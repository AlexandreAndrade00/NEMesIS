import filecmp
import os
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from random import randint, uniform
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Dict
from itertools import takewhile

import logging

from nemesis.misc.enums import (
    AttributeType,
    Entity,
    OptimiserType,
    FuseType,
)
from .phenotype import Layer, Optimiser, Fusion

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")


class Attribute(Generic[T]):
    def __init__(
        self,
        var_type: str,
        num_values: int,
        min_value: T,
        max_value: T,
        generator: Callable[[int, T, T], List[T]] | None,
    ) -> None:
        self.var_type = var_type
        self.num_values: int = num_values
        self.min_value: T = min_value
        self.max_value: T = max_value
        self.generator: Callable[[int, T, T], List[T]] | None = generator
        self.values: Optional[List[T]] = None

    @classmethod
    def from_values(cls, var_type: str, num_values: int, min_value: T, max_value: T, values: List[T]) -> "Attribute":
        attr = cls(var_type, num_values, min_value, max_value, None)

        attr.values = values

        return attr

    def generate(self) -> None:
        if self.generator is None:
            raise RuntimeError("Attribute doesn't have a generator")

        self.values = self.generator(self.num_values, self.min_value, self.max_value)
        assert self.values is not None

    def __repr__(self) -> str:
        return (
            f"Attribute(num_values={self.num_values},"
            + f" min_value={self.min_value},"
            + f" max_value={self.max_value},"
            + f" values={self.values})"
        )

    def __str__(self) -> str:
        string: str = f"{self.var_type},{self.num_values},{self.min_value},{self.max_value}"
        if self.values is not None:
            string += f",{self.values}"
        return string

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Attribute):
            return (
                self.var_type == other.var_type
                and self.num_values == other.num_values
                and self.min_value == other.min_value
                and self.max_value == other.max_value
                and self.values == other.values
            )
        return False


@dataclass
class Symbol:
    name: str

    @staticmethod
    def create_symbol(symbol_str: str) -> "Symbol":
        symbol_name: str
        attribute_type: str
        num_values: str
        min_value: str
        max_value: str
        if "<" in symbol_str and ">" in symbol_str:
            symbol_name = symbol_str.replace("<", "").replace(">", "").rstrip().lstrip()
            return NonTerminal(symbol_name)
        if "[" in symbol_str and "]" in symbol_str:
            attribute: Attribute
            symbol_name, attribute_type, num_values, min_value, max_value = (
                symbol_str.replace("[", "").replace("]", "").split(",")
            )

            if AttributeType(attribute_type) == AttributeType.INT:
                attribute = Attribute[int](
                    attribute_type,
                    int(num_values),
                    int(min_value),
                    int(max_value),
                    lambda n, min, max: [randint(min, max) for _ in range(n)],
                )
            elif AttributeType(attribute_type) == AttributeType.FLOAT:
                attribute = Attribute[float](
                    attribute_type,
                    int(num_values),
                    float(min_value),
                    float(max_value),
                    lambda n, min, max: [uniform(min, max) for _ in range(n)],
                )
            elif AttributeType(attribute_type) == AttributeType.INT_POWER2:
                attribute = Attribute[int](
                    attribute_type,
                    int(num_values),
                    int(min_value),
                    int(max_value),
                    lambda n, min, max: [2 ** randint(min, max) for _ in range(n)],
                )
            elif AttributeType(attribute_type) == AttributeType.INT_POWER2_INV:
                attribute = Attribute[int](
                    attribute_type,
                    int(num_values),
                    int(min_value),
                    int(max_value),
                    lambda n, min, max: [1 / (2 ** randint(min, max)) for _ in range(n)],
                )
            else:
                raise AttributeError(f"Invalid Attribute type: [{attribute_type}]")
            return Terminal(symbol_name, attribute)
        return Terminal(symbol_str)

    def __lt__(self, other: "Symbol") -> bool:
        return self.name < other.name

    def __le__(self, other: "Symbol") -> bool:
        return self.name <= other.name

    def __gt__(self, other: "Symbol") -> bool:
        return self.name > other.name

    def __ge__(self, other: "Symbol") -> bool:
        return self.name >= other.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Symbol):
            return self.__dict__ == other.__dict__
        return False

    def __hash__(self) -> int:
        return hash(repr(self))


class NonTerminal(Symbol):
    def __str__(self) -> str:
        return f"<{self.name}>"


@dataclass
class Terminal(Symbol):
    attribute: Optional["Attribute"] = field(default=None)

    def __hash__(self) -> int:
        return hash(repr(self))

    def __str__(self) -> str:
        if self.attribute is None:
            return self.name
        return f"[{self.name},{str(self.attribute)}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Terminal):
            return self.__dict__ == other.__dict__
        return False


@dataclass
class Genotype:
    expansions: dict[NonTerminal, list[List[Symbol]]]
    codons: dict[Symbol, list[int]]

    @classmethod
    def empty(cls) -> "Genotype":
        return cls({}, {})

    def _concatenate_to_dict(
        self,
        dict: dict[K, list[T]],  # pylint: disable=redefined-builtin
        key: K,
        element: T,
        mode: str = "append",
    ) -> dict[K, list[T]]:
        if key not in dict.keys():
            dict[key] = [element]
        else:
            if mode == "append":
                dict[key] = dict[key] + [element]
            elif mode == "prepend":
                dict[key] = [element] + dict[key]
            else:
                raise ValueError(f"Unrecognised value: [{mode}]. Only 'append' and 'prepend are accepted")
        return dict

    def add_to_genome(self, non_terminal: NonTerminal, codon: int, derivation: List[Symbol], mode: str) -> None:
        self.codons = self._concatenate_to_dict(self.codons, non_terminal, codon, mode)
        self.expansions = self._concatenate_to_dict(self.expansions, non_terminal, derivation, mode)

    def __iadd__(self, other: "Genotype") -> "Genotype":
        for k in other.expansions.keys():
            if k not in self.expansions.keys():
                self.expansions[k] = other.expansions[k]
            else:
                self.expansions[k] += other.expansions[k]

        for i in other.codons.keys():
            if i not in self.codons.keys():
                self.codons[i] = other.codons[i]
            else:
                self.codons[i] += other.codons[i]

        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Genotype):
            return self.__dict__ == other.__dict__
        return False


class Grammar:
    def __init__(self, path: str, backup_path: Optional[str] = None):
        self.grammar = self.get_grammar(path)
        if backup_path is not None:
            self._backup_used_grammar(path, backup_path)

    def _backup_used_grammar(self, origin_filepath: str, destination: str) -> None:
        destination_filepath: str = os.path.join(destination, "used_grammar.grammar")
        # if there is a config file backed up already and it is different than the one we are trying to backup
        if os.path.isfile(destination_filepath) and filecmp.cmp(origin_filepath, destination_filepath) is False:
            raise ValueError(
                "You are probably trying to continue an experiment "
                "with a different grammar than the one you used initially. "
                "This is a gentle reminder to double-check the grammar you "
                "have just passed as parameter."
            )
        # pylint: disable=protected-access
        if not shutil._samefile(origin_filepath, destination_filepath):  # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)

    def get_grammar(self, path: str) -> dict[NonTerminal, list[List[Symbol]]]:
        raw_grammar: Optional[list[str]] = self.read_grammar(path)

        if raw_grammar is None:
            logger.error("Grammar file does not exist.")
            sys.exit(-1)

        return self.parse_grammar(raw_grammar)

    def read_grammar(self, path: str) -> Optional[list[str]]:
        try:
            with open(path, "r", encoding="utf-8") as f_in:
                raw_grammar = f_in.readlines()
                return raw_grammar
        except IOError:
            return None

    def parse_grammar(self, raw_grammar: list[str]) -> dict[NonTerminal, list[List[Symbol]]]:
        grammar = {}
        for rule in raw_grammar:
            non_terminal_name, raw_rule_expansions = rule.rstrip("\n").split("::=")
            nt_symbol: Symbol = Symbol.create_symbol(non_terminal_name)
            assert isinstance(nt_symbol, NonTerminal)

            rule_expansions: list[List[Symbol]] = []
            for production_rule in raw_rule_expansions.split("|"):
                rule_expansions.append(
                    [Symbol.create_symbol(symbol_name) for symbol_name in production_rule.rstrip().lstrip().split(" ")]
                )
            grammar[nt_symbol] = rule_expansions
        return grammar

    def __str__(self) -> str:
        print_str = ""

        for _key_ in sorted(self.grammar):
            production_list: list[str] = []

            for production in self.grammar[_key_]:
                symbols: list[str] = [str(symbol) for symbol in production]
                production_list.append(" ".join(symbols))

            print_str += f"{str(_key_)} ::= {' | '.join(production_list)}\n"

        return print_str

    def initialise(self, start_symbol_name: str) -> Genotype:
        start_symbol: NonTerminal = NonTerminal(start_symbol_name)

        genotype: Genotype = self.initialise_recursive(start_symbol)

        return genotype

    def initialise_recursive(self, symbol_to_expand: Symbol) -> Genotype:
        genotype: Genotype = Genotype(expansions={}, codons={})

        if isinstance(symbol_to_expand, NonTerminal):
            expansion_possibility: int = randint(0, len(self.grammar[symbol_to_expand]) - 1)

            derivation: List[Symbol] = deepcopy(self.grammar[symbol_to_expand][expansion_possibility])

            for expanded_symbol in derivation:
                if isinstance(expanded_symbol, Terminal) and expanded_symbol.attribute is not None:
                    assert expanded_symbol.attribute.values is None
                    # this method has side-effects. The List[Symbol] object is altered because of this
                    expanded_symbol.attribute.generate()

                genotype += self.initialise_recursive(expanded_symbol)

            genotype.add_to_genome(symbol_to_expand, expansion_possibility, derivation, mode="prepend")

        return genotype

    def decode(self, start_symbol_name: str, genotype: Genotype) -> Layer | Fusion | Optimiser:
        start_symbol: NonTerminal = NonTerminal(start_symbol_name)

        phenotype_tokens: list[str]

        unconsumed_genotype: Genotype = deepcopy(genotype)

        # to keep track of any extra codons/expansions that were used
        extra_genotype: Genotype = Genotype.empty()

        phenotype_tokens = self._decode_recursive(start_symbol, unconsumed_genotype, extra_genotype)
        # if we decoded an individual that has suffered a DSGE mutation we will update the genotype accordingly
        # therefore, we will remove unconsumed codons/expansions and add extra codons/expansions that were used

        for k in unconsumed_genotype.expansions.keys():
            n_expansions: int = len(genotype.expansions[k])
            n_unconsumed_expansions: int = len(unconsumed_genotype.expansions[k])
            genotype.expansions[k] = genotype.expansions[k][: n_expansions - n_unconsumed_expansions]
            genotype.codons[k] = genotype.codons[k][: n_expansions - n_unconsumed_expansions]

            if k in extra_genotype.expansions.keys():
                genotype.expansions[k] += extra_genotype.expansions[k]
            if k in extra_genotype.codons.keys():
                genotype.codons[k] += extra_genotype.codons[k]

            if not genotype.expansions[k]:
                genotype.expansions.pop(k)
            if not genotype.codons[k]:
                genotype.codons.pop(k)

        for k in extra_genotype.expansions.keys():
            if k not in genotype.expansions.keys():
                genotype.expansions[k] = extra_genotype.expansions[k]
                genotype.codons[k] = extra_genotype.codons[k]

        phenotype: str = " ".join(phenotype_tokens)

        return self._parse_phenotype(phenotype)

    def _decode_recursive(self, symbol: Symbol, unconsumed_geno: Genotype, extra_genotype: Genotype) -> list[str]:
        phenotype: list[str] = []

        if isinstance(symbol, NonTerminal):
            # consume expansion
            expansion: Optional[List[Symbol]] = None

            if symbol in unconsumed_geno.expansions.keys() and len(unconsumed_geno.expansions[symbol]) > 0:
                expansion = unconsumed_geno.expansions[symbol].pop(0)

            # In case there has been a DSGE mutation, a symbol might not have enough codons
            # to continue expanding, thus throwing an error
            if expansion is None:
                expansion_possibility: int = randint(0, len(self.grammar[symbol]) - 1)

                expansion = deepcopy(self.grammar[symbol][expansion_possibility])

                extra_genotype.add_to_genome(symbol, expansion_possibility, expansion, mode="append")

            for expanded_symbol in expansion:
                phenotype += self._decode_recursive(expanded_symbol, unconsumed_geno, extra_genotype)

        elif isinstance(symbol, Terminal):
            if symbol.attribute is None:
                return [f"{symbol.name}"]

            if symbol.attribute.values is None:
                symbol.attribute.generate()

            assert symbol.attribute.values is not None

            return [f"{symbol.name}:{','.join(map(str, symbol.attribute.values))}"]
        else:
            raise ValueError(f"Unknown symbol: {symbol}")

        return phenotype

    def _parse_phenotype(self, phenotype: str) -> Layer | Fusion | Optimiser:
        # ignore modules separator
        phenotype = phenotype.replace("| ", "")
        phenotype_as_list: List[List[str]] = list(map(lambda x: x.split(":"), phenotype.split(" ")))

        result: Layer | Optimiser | Fusion

        entity_info: List[str] = phenotype_as_list.pop(0)

        entity: Entity = Entity(entity_info[0])

        name: str = entity_info[1]

        entity_parameters: Dict[str, str] = {
            kv[0]: kv[1] for kv in takewhile(lambda kv: kv[0] not in Entity.enum_values(), phenotype_as_list)
        }

        if entity == Entity.LAYER:
            result = Layer(
                type=name,
                parameters=entity_parameters,
            )

        elif entity == Entity.OPTIMISER:
            result = Optimiser(optimiser_type=OptimiserType(name), optimiser_parameters=entity_parameters)
        elif entity == Entity.FUSION:
            result = Fusion(FuseType.from_string(name), entity_parameters)
        else:
            raise ValueError(f"Unknown entity: {entity}")

        return result

    def search_symbol(self, query: str, query_space: Symbol | List[Symbol]) -> list[Tuple[int, Symbol]]:
        if isinstance(query_space, NonTerminal):
            for i, expansions in enumerate(self.grammar[query_space]):
                if (s := self.search_symbol(query, expansions)) is not None:
                    return [(i, query_space), *s]
        elif isinstance(query_space, Terminal):
            if query_space.name == query:
                return []
        elif isinstance(query_space, List):
            for i, symbol in enumerate(query_space):
                assert isinstance(symbol, Symbol)

                if (s := self.search_symbol(query, symbol)) is not None:
                    return [(i, symbol), *s]

        return []
