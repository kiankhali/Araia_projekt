import duckdb
import pandas

# Verbindung zur DuckDB-Datenbank herstellen (im Speicher)
conn = duckdb.connect(database=':memory:')
conn.execute('''
CREATE TABLE OP(
  Dringlichkeit VARCHAR NOT NULL,
  Startzeit TIME NOT NULL,
  Endzeit TIME NOT NULL,
  CHECK (Dringlichkeit IN ('Notoperation', 'dringliche Operation', 'frühelektive Operation', 'elektive Operation' )),
             CHECK ((EXTRACT('hour' FROM Endzeit)*60)+EXTRACT('minute' FROM Endzeit)-((EXTRACT('hour' FROM Startzeit)*60)+EXTRACT('minute' FROM Startzeit))>=15),
  CHECK ((EXTRACT('hour' FROM Endzeit)*60)+EXTRACT('minute' FROM Endzeit)-((EXTRACT('hour' FROM Startzeit)*60)+EXTRACT('minute' FROM Startzeit))<=300)
  
)
''')

conn.execute("INSERT INTO OP (Dringlichkeit, Startzeit, Endzeit) VALUES ('frühelektive Operation', '15:30:00.123456', '15:44:00.123456')")

# Zeitunterschied in Minuten abfragen
result = conn.execute("SELECT ((EXTRACT('hour' FROM Endzeit)*60)+EXTRACT('minute' FROM Endzeit))-((EXTRACT('hour' FROM Startzeit)*60)+EXTRACT('minute' FROM Startzeit)) AS Zeitunterschied FROM OP")
print(result.fetch_df())
