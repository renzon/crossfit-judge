# crossfit-judge
Project to play with computer vision to judge crossfit exercises


## Intalling

Install any python 3 version and create a virtualenv:

```bash
python -m venv .venv
```

Activate virtualenv
```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

To avoid leaking my API key on version control, I used [Python Decouple](https://pypi.org/project/python-decouple/).
So you can crete a `.env` file, which will be ignored on git, by running:

```bash
cp env-sample .env
```

You should add your key `.env` to be able to download your models.



First I (Renzo Nuccitelli) started by studying basic concepts. Check [Learning Readme](./learning/README.md) for details on the study.
