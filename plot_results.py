import json
import matplotlib.pyplot as plt

with open("results/metrics.json", "r") as f:
    history = json.load(f)

rounds = [r["round"] for r in history]
loss = [r["loss"] for r in history]
accuracy = [r["accuracy"] for r in history]
trust = [r["trust_score"] for r in history]

plt.figure()
plt.plot(rounds, loss, marker="o")
plt.title("Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.savefig("results/loss.png")
plt.close()

plt.figure()
plt.plot(rounds, accuracy, marker="o")
plt.title("Accuracy per Round")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.savefig("results/accuracy.png")
plt.close()

plt.figure()
plt.plot(rounds, trust, marker="o")
plt.title("Trust Score per Round")
plt.xlabel("Round")
plt.ylabel("Trust Score")
plt.savefig("results/trust.png")
plt.close()

print("Plots saved in results/")
