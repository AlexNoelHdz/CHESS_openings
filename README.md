<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p>

<h3 align="center">Chess openings active learning system.</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://www.mit.edu/~amini/LICENSE.md)

</div>

---

<p align="center">
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [Project structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Authors](#authors)

## 🧐 About <a name = "about"></a>

In the learning stage to play chess, learning chess openings is essential to aspire to be an intermediate level player and this can be overwhelming for a beginner level person because reading chess books is a complicated task because they are divided in sections where they explain opening by opening and movement by movement (there are at least 1,300 opening variants each with up to 21 movements), in addition, the web resources and mobile applications on the market present limitations such as that they are generally paid or are designed for passive study.

This project is a didactic learning system where you can play against a computer that plays openings randomly and gives you constant position feedback.

## 🎋 Project structure <a name = "project-structure"></a>
- +---0.TOG:      Degree obtaining project
- +---data:       Folder to store the used or generated data sets.
- +---notebooks:  Jupyter Notebooks for exploratory analysis and data processing.
- +---pickles/
- ª   +---models: Trained models and their metadata.
- +---stockfish/: Stockfish artificial intelligence comes in as an auxiliary once the opening phase passes.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

```bash
git clone https://github.com/AlexNoelHdz/CHESS_openings
cd CHESS_openings
pip install -r requirements.txt
```

## 🎈 Usage <a name="usage"></a>

After cloning project just run
```
py .\play_chess.py
```

## ✍️ Authors <a name = "authors"></a>

- [@AlexNoelHdz](https://github.com/AlexNoelHdz)

