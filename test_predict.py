"""Send a sample student to the running service.

Usage:
    python test_predict.py            # local service on port 9696
    python test_predict.py <url>      # any other deployment
"""

import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

student = {
    "school": "MS",
    "sex": "M",
    "age": 18,
    "address": "R",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 1,
    "Fedu": 1,
    "Mjob": "at_home",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 1,
    "failures": 2,
    "schoolsup": "no",
    "famsup": "no",
    "paid": "no",
    "activities": "no",
    "nursery": "yes",
    "higher": "no",
    "internet": "no",
    "romantic": "yes",
    "famrel": 3,
    "freetime": 4,
    "goout": 5,
    "Dalc": 3,
    "Walc": 4,
    "health": 3,
    "absences": 10,
}

response = requests.post(url, json=student, timeout=10)
print(response.json())
