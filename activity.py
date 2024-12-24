import os

from mpi4py import MPI
from mesa import Agent, Model
from mesa.time import BaseScheduler
from concurrent.futures import ThreadPoolExecutor
import random
import cv2
from rdflib import Graph, URIRef, Literal, RDF
from rdflib.namespace import XSD
from imutils import contours

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

GALLE_RANGE = range(1, 11)
HAMBANTOTA_RANGE = range(11, 15)
MONERAGALA_RANGE = range(15, 18)

ontology_path = r"C:\Users\User\Desktop\CSAT\VoteAgent.owl"
graph = Graph()
graph.parse(ontology_path, format="xml")




num_datasets = 17
parties = ["PartyA", "PartyB", "PartyC", "PartyD", "PartyE", "PartyF"]


class CameraAgent(Agent):
    def __init__(self, unique_id, model, dataset_path, counting_agent):
        super().__init__(unique_id, model)
        self.dataset_path = dataset_path
        self.counting_agent = counting_agent


    def process_image(self, image_path):

        image = cv2.imread(image_path)
        dim = (1398, 1783)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50:
                cv2.drawContours(thresh, [c], -1, 0, -1)

        invert = 255 - thresh
        offset, old_cY, old_cX, first = 10, 0, 0, True
        cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

        crosscount = -1
        table = []
        row = []
        prevX = 0
        current_row_index = -1
        column_centroids = []
        value = 0
        value1 = 0
        processed_centroids = []
        size = 0
        party = None
        terminate_main_loop = False

        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if 0 <= cX <= 1000 or 0 <= cY <= 100:
                    continue
                too_close = False
                for (prev_cX, prev_cY) in processed_centroids:
                    if abs(cX - prev_cX) < 200 and abs(cY - prev_cY) < 200:
                        too_close = True
                        break
                if too_close:
                    continue
                processed_centroids.append((cX, cY))
                if abs(cY - old_cY) > offset:
                    if not first:
                        table.append(row)
                    old_cY = cY
                    row = []
                    current_row_index += 1
                column_index = -1
                for i, column_cX in enumerate(column_centroids):
                    if abs(cX - column_cX) < 50:
                        column_index = i
                        break

                if column_index == -1:
                    column_index = len(column_centroids)
                    column_centroids.append(cX)
                if column_index == 0:
                    x, y, w, h = cv2.boundingRect(c)
                    roi = invert[y:y + h, x:x + w]
                    crosscount += 1
                    roi_cnts = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    roi_cnts = roi_cnts[0] if len(roi_cnts) == 2 else roi_cnts[1]
                    for roi_c in roi_cnts:
                        epsilon = 0.04 * cv2.arcLength(roi_c, True)
                        approx = cv2.approxPolyDP(roi_c, epsilon, True)
                        num_vertices = len(approx)
                        if num_vertices > 4:
                            if value < 2:
                                if num_vertices == 8:
                                    if prevX == 0:
                                        value += 1
                                        prevX = cX
                                        value1 = crosscount
                                        size = len(approx)
                                    else:
                                        if prevX == cX:
                                            value -= 1
                                            value1 = crosscount
                                            size = len(approx)
                                            value += 1
                                        else:
                                            value += 1
                                else:
                                    if prevX != cX:
                                        value += 1
                                        prevX = cX
                            else:
                                terminate_main_loop = True
                                break
                if terminate_main_loop:
                    break

        if value == 1:
            if size == 8:
                if value1 == 0:
                    party = "PartyA"
                elif value1 == 1:
                    party = "PartyB"
                elif value1 == 2:
                    party = "PartyC"
                elif value1 == 3:
                    party = "PartyD"
                elif value1 == 4:
                    party = "PartyE"
                else:
                    party = "PartyF"
        if party:
            return party

    def process_images(self):
        all_images = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(subdir, file))

        random.shuffle(all_images)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.process_image, all_images))


        for party in results:
            if party:
                self.counting_agent.receive_vote(party)


class CountingAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.vote_count = {party: 0 for party in parties}
    def receive_vote(self, party):
        if party in self.vote_count:
            self.vote_count[party] += 1
    def get_count(self):
        return self.vote_count
class Polling_Division_Model(Model):
    def __init__(self, dataset_path):
        self.schedule = BaseScheduler(self)
        self.dataset_path = dataset_path

        self.counting_agent = CountingAgent(2,self)
        self.schedule.add(self.counting_agent)

        self.camera_agent = CameraAgent(1, self, self.dataset_path, self.counting_agent)
        self.schedule.add(self.camera_agent)
    def step(self):
        self.camera_agent.process_images()
        return self.counting_agent.get_count()


def update_ontology(district, dataset_id, votes):
    try:
        BASE_NAMESPACE = "http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#"
        agent_uri = URIRef(BASE_NAMESPACE + f"PollingAgent{dataset_id}")
        if (agent_uri, RDF.type, None) in graph:
            for party, count in votes.items():
                vote_property = URIRef(BASE_NAMESPACE + f"Votecount{party}")
                graph.set((agent_uri, vote_property, Literal(count, datatype=XSD.integer)))
        else:
            raise ValueError(f"PollingAgent{dataset_id} not found.")
    except Exception as e:
        print(f"Error updating ontology for {district} and dataset {dataset_id}: {e}")

def update_District_results(district,party,total_votes):
    BASE_NAMESPACE = "http://www.semanticweb.org/user/ontologies/2024/11/untitled-ontology-6#"

    try:
        district_uri = URIRef(BASE_NAMESPACE + district)
        if (district_uri, RDF.type, None) in graph:
            finalvoteproperty = URIRef(BASE_NAMESPACE + f"TotalCount{party.replace('Party', '')}")
            print(f"Updating {district} TotalCount{party.replace('Party', '')}: Total votes{total_votes}")
            graph.set((district_uri, finalvoteproperty, Literal(total_votes, datatype=XSD.integer)))
        else:
            print(f"District {district}Agent not found in the graph.")
        print("Ontology updated and saved.")
    except Exception as e:
        print(f"Error updating ontology: {e}")

if __name__ == "__main__":
    dataset_root = r"C:\Users\User\Desktop\New Votes DataSet"
    num_datasets = 17
    if rank == 0:
        district_results = {
            "GalleDistrictAgent": {},
            "HambantotaDistrictAgent": {},
            "MoneragalaDistrictAgent": {}
        }
        district_totals = {
            "GalleAgent": {party: 0 for party in parties},
            "HambantotaAgent": {party: 0 for party in parties},
            "MonaragalaAgent": {party: 0 for party in parties}
        }
        for i in range(1, size):
            if i in GALLE_RANGE:
                result = comm.recv(source=i, tag=i)
                district_results["GalleDistrictAgent"][f"Dataset_{i}"] = result
            elif i in HAMBANTOTA_RANGE:
                result = comm.recv(source=i, tag=i)
                district_results["HambantotaDistrictAgent"][f"Dataset_{i}"] = result
            elif i in MONERAGALA_RANGE:
                result = comm.recv(source=i, tag=i)
                district_results["MoneragalaDistrictAgent"][f"Dataset_{i}"] = result
            for party, count in result.items():
                if i in GALLE_RANGE:
                    district_totals["GalleAgent"][party] += count
                elif i in HAMBANTOTA_RANGE:
                    district_totals["HambantotaAgent"][party] += count
                elif i in MONERAGALA_RANGE:
                    district_totals["MonaragalaAgent"][party] += count
            update_ontology("DistrictAgent", i, result)
        for district, totals in district_totals.items():
            for party, total_votes in totals.items():
                update_District_results(district,party,total_votes)

        print("\nFinal Results:")
        for district, datasets in district_results.items():
            print(f"\n{district}:")
            for dataset, votes in datasets.items():
                print(f"  {dataset}: {votes}")
        # graph.serialize(destination="updated_ontology2.owl", format="xml")

    else:
        dataset_id = rank
        if dataset_id <= num_datasets:
            dataset_path = os.path.join(dataset_root, f"Dataset_{dataset_id}")
            Division_model = Polling_Division_Model(dataset_path)
            results = Division_model.step()
            comm.send(results, dest=0, tag=dataset_id)
