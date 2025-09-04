FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /usr/local/app

COPY nemesis nemesis
COPY settings settings
COPY scripts scripts
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod 755 scripts/run.sh

ENTRYPOINT ["./scripts/run.sh"]
