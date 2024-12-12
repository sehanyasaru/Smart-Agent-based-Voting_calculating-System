Overview
This system is designed to automate the vote counting process in Sri Lanka's elections using real-time camera integration for ballot image recognition. The system uses distributed agents to process images of ballots, count votes, and update a central database for real-time results.

Key Features:
Mobile Integration: Capture ballot images using a mobile camera and send them to the server.
Machine Learning Model: Use a trained model to recognize the party and "X" sign on the ballot.
Distributed Architecture: The system leverages MPI (Message Passing Interface) for concurrent processing across multiple agents.
Real-time Voting Results: Aggregate results are stored and updated in a Firebase database, allowing real-time tracking of election results.
Components
1. Camera Agents:
Capture ballot images using a mobile device and upload them to the system.
Use an ML model to detect the "X" sign on the ballot and identify the party.
2. Counting Agents:
Receive processed vote data from Camera Agents.
Maintain local buffers for vote counts for each party and update results based on incoming data.
3. Polling Division Agents:
Aggregate votes from Counting Agents within a district.
Maintain RDF data stores and prepare aggregated results for district-level reporting.
4. District Agents:
Collect results from Polling Division Agents.
Update Firebase for real-time visualization on the web interface.
Requirements
Python 3.6+
Libraries:
tensorflow (for ML model inference)
mpi4py (for parallel processing)
Flask (for receiving images from mobile camera)
opencv-python (for image processing)
requests (for network communication)
firebase-admin (for Firebase integration)
