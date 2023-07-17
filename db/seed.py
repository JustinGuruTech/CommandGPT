import sqlite3

# Establish a database connection
conn = sqlite3.connect('fielddata.db')
c = conn.cursor()

# Create tables
c.execute('''
CREATE TABLE IF NOT EXISTS Job (
    id INTEGER PRIMARY KEY,
    description TEXT,
    location TEXT,
    date DATE,
    hours INTEGER,
    notes TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS Crew (
    id INTEGER PRIMARY KEY,
    name TEXT,
    hours INTEGER
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS Materials (
    id INTEGER PRIMARY KEY,
    name TEXT,
    quantity INTEGER,
    unit TEXT
)
''')

# Commit the changes
conn.commit()

# Insert seed data into tables
jobs = [
    (1, "Worked on the house", "Jackson", "2023-07-18", 5, "Nathan didn't clock in"),
    (2, "Installed wiring", "Los Angeles", "2023-07-19", 8, "John worked overtime"),
]

crew = [
    (1, "Nathan", 5),
    (2, "John", 8),
]

materials = [
    (1, "Cat5 data", 125, "ft"),
    (2, "standard outlets", 25, "pcs"),
    (3, "GFCI outlets", 8, "pcs"),
    (4, "outlet covers", 33, "pcs"), # assume we used 33 outlet covers (25 standard + 8 GFCI)
]

# Insert the data into the tables
c.executemany('INSERT INTO Job VALUES (?, ?, ?, ?, ?, ?)', jobs)
c.executemany('INSERT INTO Crew VALUES (?, ?, ?)', crew)
c.executemany('INSERT INTO Materials VALUES (?, ?, ?, ?)', materials)

# Commit the changes and close the connection
conn.commit()
conn.close()
