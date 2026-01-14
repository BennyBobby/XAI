import pandas as pd
import matplotlib.pyplot as plt
import os

METHOD_NAME = "Grad-CAM"
INPUT_DIR = "evaluation_result"


def generer_graphique():
    safe_name = METHOD_NAME.replace(" ", "_").replace("+", "p")
    csv_path = os.path.join(INPUT_DIR, f"pointing_game_{safe_name}.csv")

    if not os.path.exists(csv_path):
        print(
            f"Erreur : Le fichier {csv_path} n'existe pas. Lancez d'abord le benchmark."
        )
        return

    df = pd.read_csv(csv_path)

    domain = df["domain"].iloc[0] if "domain" in df.columns else "Inconnu"

    hits = df["hit"].sum()
    misses = len(df) - hits
    total = len(df)
    accuracy = (hits / total) * 100

    plt.figure(figsize=(9, 7))
    bars = plt.bar(
        ["Succès (Hit)", "Échec (Miss)"],
        [hits, misses],
        color=["#2ecc71", "#e74c3c"],
        edgecolor="black",
        alpha=0.8,
    )

    plt.title(
        f"Métrique : Pointing Game\nMéthode : {METHOD_NAME} | Domaine : {domain}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Nombre d'images", fontsize=12)
    plt.xlabel(
        f"Total images testées : {total} | Précision : {accuracy:.1f}%",
        fontsize=11,
        style="italic",
    )

    plt.ylim(0, total + (total * 0.2))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (total * 0.02),
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    output_image = os.path.join(INPUT_DIR, f"graphique_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Graphique sauvegardé : {output_image}")
    plt.show()


if __name__ == "__main__":
    generer_graphique()
