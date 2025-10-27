# YOLOv8 Object Detection and Tracking 

## Example
![Screenshot](./1.png)
![Screenshot](./2.png)



##  Description

Ce projet regroupe **deux versions** d’un système de détection et de suivi d’objets en temps réel basé sur **YOLOv8**, développé en **Python** avec **OpenCV** et **PyTorch** :

1.  **Version avec interface graphique (GUI)** — pour visualiser en direct les détections, appliquer des effets visuels, et interagir via menus.
2.  **Version sans interface (headless)** — optimisée pour systèmes embarqués, IoT, Edge Computing, ou exécution distante sans affichage.

Ces deux versions partagent le même cœur logique de détection basé sur **Ultralytics YOLOv8n (nano)**, garantissant performance et légèreté.

---

## 🚀 Fonctionnalités

### 1. Version GUI (Tkinter)
- Interface utilisateur **Tkinter** simple et responsive  
- **YOLOv8n (nano)** pour détection rapide et légère  
- **Menu d’effets visuels** : sepia, grayscale, blur, pixelate, sketch, etc.  
- **Activation sélective** de la détection (personnes, obstacles, objets)  
- Capture d’image directe depuis la caméra  
- Statut en temps réel et support multi-plateforme  

### 2. Version Headless (sans GUI)
- Détection et suivi d’objets **sans interface graphique**  
- Sortie console ou possibilité d’envoi vers serveur distant (IoT / API)  
- Idéale pour :
  - **Raspberry Pi**
  - **ESP32-CAM (flux IP)**
  - **Systèmes Edge / embarqués**
  - **Surveillance autonome**
  - **Applications IoT industrielles**

---

## 🧩 Technologies utilisées

- [Python 3.8+](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Pillow (PIL)](https://python-pillow.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)

---
## 📚 Crédits
Ce projet est une adaptation/modification du code original de [vir727](https://github.com/vir727/Gesture-Recogonition-App).
