"""Retrieve sources using a spelling the user has explicitly confirmed."""

import argparse
import json
from src.prescription.knowledge import MedicineKnowledge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed-name", required=True)
    parser.add_argument(
        "--setid", help="A DailyMed label identifier returned by a preceding lookup"
    )
    args = parser.parse_args()
    client = MedicineKnowledge()
    result = (
        client.indications(args.confirmed_name, args.setid)
        if args.setid
        else client.lookup(args.confirmed_name)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
