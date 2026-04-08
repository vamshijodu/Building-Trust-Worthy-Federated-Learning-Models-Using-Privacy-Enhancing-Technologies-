# Building Trustworthy Federated Learning Models Using Privacy Enhancing Technologies

## Project Overview

This project demonstrates a trustworthy Federated Learning (FL) system where multiple decentralized clients collaboratively train a shared machine learning model without sharing raw data.

Key technologies used:

* Federated Learning using **Flower**
* **Differential Privacy** using **Opacus**
* Result visualization using **Matplotlib**

---

## Key Features

* Privacy-preserving federated training (no raw data sharing)
* Differential Privacy
* Trustworthy global aggregation
* Clear server and client-side logs
* Automatic generation of result graphs
* Clean, professional, and examiner-friendly outputs

---

## Project Structure

```
trustworthy-fl/
│
├── client.py              # Federated client with DP
├── server.py              # Federated server with logging
├── model.py               # CNN model
├── data.py                # Data loading & client partitioning
├── robust_aggregation.py  # Robust aggregation logic
├── plot_results.py        # Result visualization
├── utils.py               # Utility helpers
│
├── results/               # Metrics & plots
│   ├── metrics.json
│   ├── loss.png
│   ├── accuracy.png
│   └── global_score.png
│
├── requirements.txt
└── README.md

## System Requirements

* Python **3.8 or higher**
* Windows
* Minimum 8 GB RAM recommended

---

## Setup Instructions

Step 1: Create virtual environment:

python -m venv venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate

Step 2: Install dependencies:

pip install -r requirements.txt

## How to Run the Project

Terminal 1 – Start Server
.\venv\Scripts\python.exe server.py

Terminal 2 – Start Client 0
.\venv\Scripts\python.exe client.py --cid 0

Terminal 3 – Start Client 1
.\venv\Scripts\python.exe client.py --cid 1

Generate Result Plots:

After training completes:

.\venv\Scripts\python.exe plot_results.py

### Generated Files

```
results/
├── accuracy.png   # Accuracy vs Federated Rounds
├── loss.png       # Loss vs Federated Rounds
└── metrics.csv    # Logged metrics per round
```

These graphs can be **directly included in the final project report** as proof of execution and performance.

---

## Privacy and Trustworthiness

* **Differential Privacy:** Ensures client data confidentiality using DP-SGD
* **Privacy Budget (ε):** Displayed after each client training round
* **Trustworthy Aggregation:** Prevents direct access to client data

Your privacy budget ε tells you how much theoretical information leakage has accumulated from DP training — and it grows with training steps and decreases with more noise.
---

## Dataset Used

* **MNIST Handwritten Digits Dataset**
* Dataset is automatically downloaded during first execution
* Data remains local to each client

---

## Academic Relevance

This project demonstrates:

* Practical implementation of Federated Learning
* Integration of Privacy Enhancing Technologies
* Privacy–utility trade-off analysis
* Real-world applicability in sensitive domains

It is suitable for:

* Final-year B.Tech major project
* Academic demonstrations and viva
* Research-oriented learning

---

## Applications

* Healthcare data analysis
* Financial systems
* Distributed IoT learning
* Privacy-sensitive AI systems

---

## Conclusion

This project successfully implements a **Trustworthy Federated Learning framework** that balances model performance with strong privacy guarantees. By integrating Differential Privacy and federated training, the system ensures secure and collaborative machine learning suitable for real-world and academic applications.