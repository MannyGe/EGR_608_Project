Team Project: Simulation-Based Decision Making with Neural
Networks
1. Objective
The goal of this project is to design and analyze a discrete-event simulation (DES) model of a
real-world system and enhance decision-making using a neural network (NN).
You will:
• Build a simulation model
• Generate data from the system
• Train a neural network
• Deploy the trained model into the simulation
• Compare rule-based vs learning-based decision policies
2. Problem Structure (Required System Design)
You must design a queueing network (Q-network) with the following structure.
Core Decision Point
Aprocess selection stage where one option must be selected from 3 to 5 candidate processes.
System Flow Requirements
After the selection:
• Each selected process leads to different downstream paths.
• Example:– Process A → 2–3 subsequent processes– Process B → 1 subsequent process– Process C → 4 subsequent processes
• These paths should be designed such that the system is balanced and the selection decision
is non-trivial.
1
Additional Structural Requirements
• There must be common upstream processes before the selection stage.
• There must be common downstream processes after the selection paths merge.
• Include at least one rework loop.
• The system must eventually exit through one or more exit points.
Problem Description and Motivation
Each team must clearly define and justify their chosen system based on the project requirements.
• Identify a real-world or realistic system that satisfies the required structure
• Explain why this problem is important and worth studying
• Describe how the system is modeled to match the given project requirements
Additionally, you must clearly specify:
• Key assumptions made in the modeling process
• Required data and how it is represented in the simulation
Note: The focus is not only on building a simulation, but also on properly formulating the problem
and justifying modeling decisions.
3. System Modeling (DES)
Your simulation model must include:
• Process selection (decision-making stage)
• Scheduling
• Resource sharing
Implementation
• Define system logic (entities, resources, routing)
• Define state variables (e.g., queue length, utilization)
• Implement a rule-based policy for process selection
4. Data Generation
You must generate training data using your simulation.
2
Input and Output Definition
• Input (Features): number of entities waiting at each selectable process
• Output (Label): total time in the system
Important Requirement for Data Quality
When running the simulation iteratively, it is not sufficient for the system to simply reach a steady
state. The generated dataset must be large and diverse enough to produce meaningful neural
network performance (e.g., reasonable prediction accuracy).
• Ensure sufficient number of samples
• Ensure variability in system states
• Avoid biased or insufficient data collection
Note: The goal is to generate a dataset that enables the NN to learn meaningful input–output
relationships.
Training vs Testing
• Generate separate training and testing datasets
• Do not use the same data for both
Simulation–Learning–Deployment Pipeline (Requirement)
Your project must follow a structured pipeline that separates data generation, model training, and
deployment:
1. Data Generation Simulation Model
• Build a simulation model specifically for generating data
• Use this model to generate datasets (features and labels)
2. Offline Neural Network Training
• Export the generated dataset
• Train the neural network offline using Python
• Evaluate model performance (e.g., prediction accuracy)
3. Deployment Simulation Model
• Build a separate simulation model for evaluation
• Deploy the trained neural network into this model
• Use the NN to make decisions during simulation
Requirement:
3
• The simulation model used for data generation must be separate from the deployment (eval
uation) simulation model
• Neural network training must be performed outside the simulation (e.g., in Python)
Purpose:
• To ensure proper separation between training and evaluation
• To mimic real-world deployment of learned models
• To avoid bias and data leakage
5. Neural Network Model
You will use a feedforward neural network.
Provided
• Neural network skeleton code
• Feature structure guidance
Requirements
• Train the NN using simulation-generated data
• Compare at least two different network architectures
• Select the best-performing model
6. Deployment in Simulation
• Replace the rule-based decision with the trained NN
• The NN takes system state (queue lengths) as input
• The NN outputs the estimated total time in the system for each candidate process
• Select the process with the smallest predicted total time
7. Performance Comparison
Compare:
• Rule-Based Policy
• Neural Network-Based Policy
Evaluation Metric
• Primary metric: Time in the system
4
Additional Metrics for Discussion
• Throughput
• Average waiting time
• Resource utilization
Requirement:
• Use time in the system as the main comparison metric
• Discuss how other metrics change between policies
8. Experimental Design
Follow the procedure shown in the lecture slides:
• Define scenarios
• Define response variables
• Run consistent experiments for both policies
9. Presentation (Instead of Report)
• Each team must give a presentation instead of submitting a report
• Each team consists of 2–3 members
• All team members must present
Presentation Content
• Problem description and motivation
• System design (Q-network)
• System modeling and assumptions
• Data generation process
• Neural network design and training
• Comparison of simulation performance between rule-based and NN-integrated policies
• Key insights and discussion
10. Deliverables
• Simulation models (Simio models)
• Neural network code
• Presentation slides
5
11. Grading Criteria
• 40%: Simulation modeling quality
• 30%: Neural network and data pipeline
• 30%: Integration, comparison, and insights
12. Key Message
The goal is not to build the most complex neural network, but to understand how
learning-based decision-making compares to rule-based logic in a simulation system