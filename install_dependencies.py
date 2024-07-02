import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Lista de bibliotecas necessárias
required_packages = ["numpy", "networkx", "matplotlib", "pulp", "seaborn", "pandas"]

# Instalar cada pacote
for package in required_packages:
    try:
        install(package)
    except subprocess.CalledProcessError:
        print(f"Error installing packege: {package}")

print("All dependencies installed.")
