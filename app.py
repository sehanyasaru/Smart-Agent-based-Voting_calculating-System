from flask import Flask, render_template
from rdflib import Graph, Namespace

app = Flask(__name__, template_folder='template')

UNTITLED = Namespace("http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#")

g = Graph()
g.parse("updated_ontology1.owl")

def get_district_agents_data():
    query = """
    PREFIX untitled: <http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#>
    SELECT ?DistrictAgent ?TotalCountA ?TotalCountB ?TotalCountC ?TotalCountD ?TotalCountE ?TotalCountF
    WHERE {
        ?DistrictAgent a untitled:DistrictAgent;
        untitled:TotalCountA ?TotalCountA;
        untitled:TotalCountB ?TotalCountB;
        untitled:TotalCountC ?TotalCountC;
        untitled:TotalCountD ?TotalCountD;
        untitled:TotalCountE ?TotalCountE;
        untitled:TotalCountF ?TotalCountF;
        
    }  
    """
    results = g.query(query)
    district_data= [
        {
            "District": str(row.DistrictAgent).replace("http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#", ""),
            "PartyA": int(row.TotalCountA),
            "PartyB": int(row.TotalCountB),
            "PartyC": int(row.TotalCountC),
            "PartyD": int(row.TotalCountD),
            "PartyE": int(row.TotalCountE),
            "PartyF": int(row.TotalCountF),
        }
        for row in results
    ]
    return district_data
def get_polling_agents_data():

    query = """
    PREFIX untitled: <http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#>
    SELECT ?PollingAgent ?VotingCenter ?VotecountPartyA ?VotecountPartyB ?VotecountPartyC ?VotecountPartyD ?VotecountPartyE ?VotecountPartyF
    WHERE {
        ?VotingCenter a untitled:VotingCenter ;
                  untitled:hasDivisionAgent ?PollingAgent .
        ?PollingAgent a untitled:PollingDivisionAgent ;
                      untitled:VotecountPartyA ?VotecountPartyA ;
                      untitled:VotecountPartyB ?VotecountPartyB ;
                      untitled:VotecountPartyC ?VotecountPartyC ;
                      untitled:VotecountPartyD ?VotecountPartyD ;
                      untitled:VotecountPartyE ?VotecountPartyE ;
                      untitled:VotecountPartyF ?VotecountPartyF .
    }
    """
    results = g.query(query)
    data = [
        {

            "VotingCenter": str(row.VotingCenter).replace("http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#", ""),
            "PartyA": int(row.VotecountPartyA),
            "PartyB": int(row.VotecountPartyB),
            "PartyC": int(row.VotecountPartyC),
            "PartyD": int(row.VotecountPartyD),
            "PartyE": int(row.VotecountPartyE),
            "PartyF": int(row.VotecountPartyF),
        }
        for row in results
    ]
    return data

@app.route("/")
def index():
    data = get_polling_agents_data()
    district_data=get_district_agents_data()
    return render_template("index1.html", polling_agents=data, district_agents=district_data)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
