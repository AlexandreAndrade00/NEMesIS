# TODO: elements inside mutation_rates should be floats instead of Any.
# I am placing type: ignore for the time being
import logging
import random
from copy import deepcopy, copy
from typing import TYPE_CHECKING, List, Tuple, Optional

from nemesis.evolution import Individual
from nemesis.evolution.grammar import Grammar, NonTerminal, Terminal
from nemesis.misc.enums import MutationType
from nemesis.config import ConfigItem
from nemesis.evolution.module import ConnectionGenotype

from uuid import UUID, uuid4

if TYPE_CHECKING:
    from nemesis.evolution.grammar import Genotype, Symbol


logger = logging.getLogger(__name__)


def mutation_add_layer(
    individual: Individual, module_name: str, grammar: Grammar, reuse_layer_prob: float, generation: int
) -> None:
    module = individual.modules[module_name]

    if len(module.layers) >= module.module_configuration.max_expansions:
        return

    is_reused = False
    if random.random() <= reuse_layer_prob and len(module.layers) > 0:
        new_layer = random.choice(list(module.layers.values()))
        is_reused = True
    else:
        new_layer = grammar.initialise(module.module_name)

    new_layer_id: UUID = uuid4()

    insert_pos: UUID = random.choice(list(module.layers.keys()))

    module.layers[new_layer_id] = new_layer

    new_layer_inputs_ids = copy(module.connections[insert_pos].direct_input_layers_ids)

    module.connections[insert_pos].direct_input_layers_ids = [new_layer_id]

    new_fusion = deepcopy(module.connections[insert_pos].fusion)

    if insert_pos == module.input_layer_id:
        module.input_layer_id = new_layer_id

        module.connections[insert_pos].fusion = grammar.initialise(module.module_configuration.fusion_rule)

    module.connections[new_layer_id] = ConnectionGenotype(new_layer_id, new_fusion, new_layer_inputs_ids, [])

    individual.track_mutation(
        MutationType.ADD_LAYER,
        generation,
        {
            "module_name": module_name,
            "layer": grammar.decode(module.module_name, new_layer),
            "reused": is_reused,
        },
    )

    logger.info(
        "Individual %d is going to have an extra layer at Module %s",
        individual.id,
        module_name,
    )


def mutation_remove_layer(individual: Individual, module_name: str, grammar: Grammar, generation: int) -> None:
    module = individual.modules[module_name]

    if len(module.layers) <= module.module_configuration.min_expansions:
        return

    module_layers_ids = copy(list(module.layers.keys()))

    module_layers_ids.remove(module.input_layer_id)

    remove_id: UUID = random.choice(module_layers_ids)

    individual.track_mutation(
        MutationType.REMOVE_LAYER,
        generation,
        {
            "module_name": module_name,
            "layer": grammar.decode(module.module_name, module.layers[remove_id]),
            "remove_id": remove_id,
        },
    )

    module.layers.pop(remove_id)

    removed_connection = module.connections.pop(remove_id)

    assert len(removed_connection.direct_input_layers_ids) == 1

    direct_connection: UUID = removed_connection.direct_input_layers_ids[0]

    if remove_id == module.output_layer_id:
        module.output_layer_id = direct_connection

    for ind_module in individual.modules.values():
        for id_con, connection in ind_module.connections.items():
            if remove_id in connection.direct_input_layers_ids:
                connection.direct_input_layers_ids.remove(remove_id)

                connection.direct_input_layers_ids.append(direct_connection)

            if remove_id in connection.skip_input_layers_ids:
                connection.skip_input_layers_ids.remove(remove_id)

    logger.info(
        "Individual %s is going to have a layer removed from Module %s; id %s",
        str(individual.id),
        module_name,
        str(remove_id),
    )


def mutation_add_connection(individual: Individual, module_name: str, layer_id: UUID, generation: int) -> None:
    module = individual.modules[module_name]

    if (
        len(module.module_configuration.skip_connections_source_modules) == 0
        or module.module_configuration.max_skip_connections == 0
    ):
        return

    if len(module.connections[layer_id].skip_input_layers_ids) >= module.module_configuration.max_skip_connections:
        removed_id = random.choice(module.connections[layer_id].skip_input_layers_ids)

        logger.info(f"Max connections reached for add connection mutation, removing random layer, id {removed_id}")

        module.connections[layer_id].skip_input_layers_ids.remove(removed_id)

    connection_skip_possibilities: List[UUID] = []

    for skip_module in module.module_configuration.skip_connections_source_modules:
        if skip_module == module_name:
            previous_layers = list(
                set(module.get_previous_layers(layer_id))
                - set(
                    module.connections[layer_id].direct_input_layers_ids
                    + module.connections[layer_id].skip_input_layers_ids
                )
            )

            if len(previous_layers) == 0:
                return

            connection_possibilities = list(
                range(
                    max(0, len(previous_layers) - module.module_configuration.self_skip_connection_max_layers_back),
                    len(previous_layers),
                )
            )

            if len(connection_possibilities) == 0:
                return

            new_input: int = random.choice(connection_possibilities)

            connection_skip_possibilities.append(previous_layers[new_input])
        else:
            connection_skip_possibilities.append(random.choice(list(individual.modules[skip_module].layers.keys())))

    # chance of being chosen is independent of number of layers or other configs
    new_connection = random.choice(connection_skip_possibilities)

    module.connections[layer_id].skip_input_layers_ids.append(new_connection)

    individual.track_mutation(
        MutationType.ADD_CONNECTION,
        generation,
        {"module_name": module_name, "layer_id": layer_id, "new_input": new_connection},
    )

    logger.info(
        "Individual %s is going to have a new connection Module %s; layer %s",
        str(individual.id),
        module_name,
        str(layer_id),
    )


def mutation_remove_connection(individual: Individual, module_name: str, layer_id: UUID, generation: int) -> None:
    module = individual.modules[module_name]

    connections = copy(module.connections[layer_id].skip_input_layers_ids)

    if len(connections) == 0:
        return

    remove_connection = random.choice(connections)

    module.connections[layer_id].skip_input_layers_ids.remove(remove_connection)

    individual.track_mutation(
        MutationType.REMOVE_CONNECTION,
        generation,
        {
            "module_name": module_name,
            "layer_id": layer_id,
            "removed_input": remove_connection,
        },
    )

    logger.info(
        "Individual %s is going to have a connection removed from Module %s; layer %s",
        str(individual.id),
        module_name,
        str(layer_id),
    )


def _mutation_dsge(gene: "Genotype", grammar: Grammar) -> None:
    nt_keys: List[NonTerminal] = sorted(list(gene.expansions.keys()))

    random_nt: NonTerminal = random.choice(nt_keys)

    nt_derivation_idx: int = random.randint(0, len(gene.expansions[random_nt]) - 1)

    nt_derivation: List[Symbol] = gene.expansions[random_nt][nt_derivation_idx]

    sge_possibilities: List[List[Symbol]] = []
    node_type_possibilities: List[type[Symbol]] = []

    if len(grammar.grammar[random_nt]) > 1:
        all_possibilities: List[Tuple[Symbol, ...]] = [tuple(derivation) for derivation in grammar.grammar[random_nt]]

        # exclude current derivation to avoid neutral mutation
        sge_possibilities = [list(d) for d in set(all_possibilities) - set([tuple(nt_derivation)])]

        node_type_possibilities.append(NonTerminal)

    terminal_symbols_with_attributes: List[Symbol] = list(
        filter(lambda x: isinstance(x, Terminal) and x.attribute is not None, nt_derivation)
    )

    if terminal_symbols_with_attributes:
        node_type_possibilities.extend([Terminal, Terminal])

    if node_type_possibilities:
        random_mt_type: type[Symbol] = random.choice(node_type_possibilities)

        if random_mt_type is Terminal:
            symbol_to_mutate: Symbol = random.choice(terminal_symbols_with_attributes)

            assert (
                isinstance(symbol_to_mutate, Terminal)
                and symbol_to_mutate.attribute is not None
                and symbol_to_mutate.attribute.values is not None
            )

            is_neutral_mutation: bool = True

            loop_interrupter = 0

            while is_neutral_mutation and loop_interrupter < 10:
                current_values = tuple(symbol_to_mutate.attribute.values)

                symbol_to_mutate.attribute.generate()

                new_values = tuple(symbol_to_mutate.attribute.values)

                if current_values != new_values:
                    is_neutral_mutation = False

                loop_interrupter += 1
        elif random_mt_type is NonTerminal:
            if len(sge_possibilities) == 0:
                logger.info("DSGE mutation didn't happen because the only possibility was neutral")

                return

            # assignment with side effect.
            # layer variable will also be affected
            new_derivation: List[Symbol] = deepcopy(random.choice(sge_possibilities))

            # this line is here because otherwise the index function
            # will not be able to find the derivation after we generate values
            gene.codons[random_nt][nt_derivation_idx] = grammar.grammar[random_nt].index(new_derivation)

            for symbol in new_derivation:
                if isinstance(symbol, Terminal) and symbol.attribute is not None:
                    assert symbol.attribute.values is None
                    symbol.attribute.generate()

            gene.expansions[random_nt][nt_derivation_idx] = new_derivation
        else:
            raise AttributeError(f"Invalid value from random_mt_type: [{random_mt_type}]")


def mutation_dgse_layer(
    individual: Individual, module_name: str, layer_id: UUID, grammar: Grammar, generation: int
) -> None:
    module = individual.modules[module_name]

    old_phenotype = grammar.decode(module.module_name, module.layers[layer_id])

    _mutation_dsge(module.layers[layer_id], grammar)

    individual.track_mutation(
        MutationType.DSGE_LAYER,
        generation,
        {"from": old_phenotype, "to": grammar.decode(module.module_name, module.layers[layer_id])},
    )

    logger.info(
        "Individual %s is going to have a DSGE mutation on Module %s; layer %s",
        str(individual.id),
        module_name,
        str(layer_id),
    )


def mutation_dgse_macro(
    individual: Individual, macro_symbol: str, macro_symbol_idx: int, grammar: Grammar, generation: int
) -> None:
    old_macro_phenotype = grammar.decode(macro_symbol, individual.macro[macro_symbol_idx])

    _mutation_dsge(individual.macro[macro_symbol_idx], grammar)

    individual.track_mutation(
        MutationType.DSGE_MACRO,
        generation,
        {"from": old_macro_phenotype, "to": grammar.decode(macro_symbol, individual.macro[macro_symbol_idx])},
    )

    logger.info(
        "Individual %s is going to have a DSGE mutation at the macro level: %s; position %d",
        str(individual.id),
        macro_symbol,
        macro_symbol_idx,
    )


def mutate(
    individual: Individual,
    grammar: Grammar,
    generation: int,
    mutation_rates: ConfigItem,
    default_train_time: float,
    max_train_time: Optional[int] = None,
) -> None:
    """
    Network mutations: add and remove layer, add and remove connections, macro structure


    Parameters
    ----------
    individual : Individual
        individual to be mutated

    grammar : Grammar
        Grammar instance, used to perform the initialisation and the genotype
        to phenotype mapping

    generation : int
        generation number at which the mutation is occurring

    mutation_rates : ConfigItem
        probabilities of each mutation type

    default_train_time : int
        default training time

    max_train_time : Optional[int]
        maximum training time for an individual
    """

    def should_mutate(mutation_rate: float) -> bool:
        return random.random() <= mutation_rate

    double_train_time: bool = False

    # in case the individual is mutated in any of the structural parameters
    # the training time is reset
    individual.total_allocated_train_time = default_train_time
    individual.reset_keys("current_time", "num_epochs", "metrics")

    for module_name, module in individual.modules.items():
        if module_name in individual.static_modules:
            continue

        # remove layer
        for _ in range(random.randint(1, 2)):
            if should_mutate(mutation_rates.remove_layer):  # type: ignore
                mutation_remove_layer(individual, module_name, grammar, generation)

        # add layer (duplicate or new)
        for _ in range(random.randint(1, 2)):
            if should_mutate(mutation_rates.add_layer):  # type: ignore
                mutation_add_layer(individual, module_name, grammar, mutation_rates.reuse_layer, generation)  # type: ignore

        # layerwise mutation
        for layer_id in module.layers.keys():
            # DSGE mutation
            if should_mutate(mutation_rates.dsge_layer):  # type: ignore
                mutation_dgse_layer(individual, module_name, layer_id, grammar, generation)

            if layer_id != UUID(int=0):
                # remove connection
                if should_mutate(mutation_rates.remove_connection):  # type: ignore
                    mutation_remove_connection(individual, module_name, layer_id, generation)

    # DSGE mutation - macro level
    for rule_idx, macro_rule in enumerate(individual.macro_rules):
        if should_mutate(mutation_rates.macro_layer):  # type: ignore
            mutation_dgse_macro(individual, macro_rule, rule_idx, grammar, generation)

    # add connections should be the last mutations
    for module_name, module in individual.modules.items():
        for layer_id in module.layers.keys():
            if layer_id != UUID(int=0) and should_mutate(mutation_rates.add_connection):  # type: ignore
                mutation_add_connection(individual, module_name, layer_id, generation)

    if double_train_time:
        individual.total_allocated_train_time *= 2
