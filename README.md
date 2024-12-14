This system is designed to automate the vote counting process in **Sri Lanka's elections** using real-time camera integration for ballot image recognition. The system uses distributed agents to process images of ballots, count votes, and update a central database for real-time results.

### Key Features:

1. **Mobile Integration:** Capture ballot images using a mobile camera and send them to the server.
2. **Machine Learning Model:** Use a trained model to recognize the party and "X" sign on the ballot.
3. **Distributed Architecture:** The system leverages MPI (Message Passing Interface) for concurrent processing across multiple agents.
4. **Real-time Voting Results:** Aggregate results are stored and updated in a Firebase database, allowing real-time tracking of election results. 

### Components

1. **Camera Agents:** 

- Capture ballot images using a mobile device and upload them to the system.
- Use an ML model to detect the "X" sign on the ballot and identify the party.

2. **Counting Agents:**

- Receive processed vote data from Camera Agents.
- Maintain local buffers for vote counts for each party and update results based on incoming data.

3.**Polling Division Agents:**

- Aggregate votes from Counting Agents within a district.
- Maintain RDF data stores and prepare aggregated results for district-level reporting.

4.**District Agents:**

- Collect results from Polling Division Agents.
- Update Firebase for real-time visualization on the web interface.

### Required Libraries

1. **TensorFlow** (for ML model inference)
2. **mpi4py** (for parallel processing)
3. **Flask** (for receiving images from mobile camera)
4. **OpenCV-python** (for image processing)
5. **requests** (for network communication)
6. **firebase-admin** (for Firebase integration)
