# shell script to run after the container is created
poetry config virtualenvs.path $PWD
poetry config virtualenvs.in-project true
poetry config virtualenvs.options.always-copy true
poetry config virtualenvs.create true
poetry self update
# Attempt to run poetry install and capture any errors
if ! poetry install; then
  # Visual reminder displayed if poetry install fails
  echo "************************************************"
  echo "**         WARNING:                           **"
  echo "**       Error starting the python env        **"
  echo "**                                            **"
  echo "************************************************"
  # Optionally, exit the script if you want to halt further execution upon failure
  # exit 1
fi
source .venv/bin/activate
echo "Done"
