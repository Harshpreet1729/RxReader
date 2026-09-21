"""Retrieve sources using a spelling the user has explicitly confirmed."""
import argparse,json
from src.prescription.knowledge import MedicineKnowledge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmed-name',required=True)
    p.add_argument('--setid',help='A DailyMed label identifier returned by a preceding lookup')
    a=p.parse_args();client=MedicineKnowledge()
    result=client.indications(a.confirmed_name,a.setid) if a.setid else client.lookup(a.confirmed_name)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
