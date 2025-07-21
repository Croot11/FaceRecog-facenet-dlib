# Smart Face Recognition Door System
A smart door access system using face recognition powered by raspberry p, designed to automatically unlock a door using facial identification. This project intergrates hardware control with machine learning techniques for real-time authetication.

## System Overview
The system is composed of:
- Camera: Captures facial images
- Raspberry pi 4: central processing unit
- Screen: Optional display for user interface
- Relay: Controls power to the door lock
- Software Stack: python, Dlib, Tensorflow, Facenet

**Workflow:**
1. The camera captures real-time video
2. Faces are detected using the Dlib library
3. Ten images per person are extrated and saved as datasets for training
4. These datasets are processed through **Facenet** to extract facial embedding vectors
5. A tensorflow classifier is trained on these embeddings
6. Once a face is detected, it is passed through the classifier
7. If recognized, Raspberry pi triggers the relay to unlock the door
