from tkinter import Tk
from Database import Database
from tkinter.filedialog import askopenfilename
from Document import Document


def start_shjit():
    database = Database('C:\\Users\\kolya\\PycharmProjects\\lang3\\Database')
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
    doc = Document(database, filename)
    doc.calculates_weights()
    print("\nClassic referat")
    print(doc.generate_classic())
    print("\nWord referat")
    print(doc.generate_words_referat())
    print(filename)


if __name__ == '__main__':
    start_shjit()
