def generate_schema(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Job)")
    columns = cursor.fetchall()
    schema = {
        "Job": {
            "fields": {}
        }
    }
    for column in columns:
        name = column[1]
        type = column[2]
        schema["Job"]["fields"][name] = type
    return schema
