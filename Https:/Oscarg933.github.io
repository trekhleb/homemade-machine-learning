language: python
python:
  - "3.6"

# Install dependencies.
install:
  - pip install -r requirements.txt

# Run linting and tests.
script:
  - pylint ./homemade

# Turn email notifications off.
notifications:
  email: false
