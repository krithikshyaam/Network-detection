# Network-detection
AI-Based Network Intrusion Detection System (Student Project)
This project demonstrates how to use Machine Learning (Random Forest) and Generative AI (Grok) to detect and explain network attacks (specifically DDoS).

How to Use
Enter API Key: Paste your Grok API key in the sidebar (optional, for AI explanations).
Train Model: Click the "Train AI Model" button. The system loads the Friday-WorkingHours... dataset automatically.
Simulate: Click "Simulate Random Packet" to pick a real network packet from the test set.
Analyze: See if the model flags it as BENIGN or DDoS, and ask Grok to explain why.

 Files
app.py: The main Python application code.
requirements.txt: List of libraries used.
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv: The dataset (CIC-IDS2017 subset).
 About
Created for a university cybersecurity project to demonstrate the integration of traditional ML and LLMs in security operations.
