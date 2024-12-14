from mpi4py import MPI
from mesa import Agent, Model
from mesa.time import BaseScheduler
import os
import random
import cv2
from imutils import contours

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class DatasetAgent(Agent):
    def __init__(self, unique_id, model, dataset_path):
        super().__init__(unique_id, model)
        self.dataset_path = dataset_path
        self.local_vote_counts = {
            "PartyA": 0,
            "PartyB": 0,
            "PartyC": 0,
            "PartyD": 0,
            "PartyE": 0,
            "PartyF": 0
        }

    def process_images(self):
        # Gather all images in the dataset and shuffle
        all_images = []
        for subdir, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    all_images.append(os.path.join(subdir, file))

        random.shuffle(all_images)  # Shuffle images

        # Process each shuffled image
        for image_path in all_images:
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
            current_column_index = -1
            column_centroids = []
            value = 0
            value1 = 0
            approxval = None
            processed_centroids = []
            outx = 0
            outy = 0
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
                                            approxval = approx
                                            size = len(approx)
                                            outx = x
                                            outy = y
                                        else:
                                            if prevX == cX:
                                                value -= 1
                                                value1 = crosscount
                                                approxval = approx
                                                size = len(approx)
                                                outx = x
                                                outy = y
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
                    self.local_vote_counts[party] += 1
        return self.local_vote_counts

    def step(self):
        self.process_images()



class DatasetModel(Model):
    def __init__(self, dataset_path):
        self.schedule = BaseScheduler(self)
        self.dataset_path = dataset_path


        agent = DatasetAgent(1, self, self.dataset_path)
        self.schedule.add(agent)

    def step(self):
        self.schedule.step()


# Main MPI + Mesa Execution
if __name__ == "__main__":
    dataset_root = "C:\\Users\\User\\Desktop\\New Votes DataSet"
    num_datasets = 17

    if rank == 0:  #polling agent
      
        results = {}


        for i in range(1, size):
            if i <= num_datasets:
                result = comm.recv(source=i, tag=i)
                results[f"Dataset_{i}"] = result

        # Print aggregated results
        print("Final Results:")
        for dataset, votes in results.items():
            print(f"{dataset}: {votes}")

    else:

        dataset_id = rank
        if dataset_id <= num_datasets:
            dataset_path = os.path.join(dataset_root, f"Dataset_{dataset_id}")
            model = DatasetModel(dataset_path)
            model.step()
            result = model.schedule.agents[0].local_vote_counts
            comm.send(result, dest=0, tag=dataset_id)
