from models import Base
from session import engine

Base.metadata.create_all(bind=engine)
print("Tabels have been created!")