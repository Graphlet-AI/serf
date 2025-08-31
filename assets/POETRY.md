# Install Poetry

To install Poetry (see [install docs](https://python-poetry.org/docs/)), you can use [pipx](https://github.com/pypa/pipx) or `curl` a script. Note that `curl` is sometimes blocked by firewalls.

## Install Poetry Using `curl`

To install using `curl`, just run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Install Poetry using `pipx`

To first install `pipx`, on Mac OS X run:

```bash
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
```

On Ubuntu Linux, run:

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
```

Then install Poetry using `pipx`:

```bash
pipx install poetry
```

It will now be outside your virtual environment and available to all your Python projects.
