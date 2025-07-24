from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import DateTime

Base = declarative_base()

class Detection(Base):
  __tablename__ = "detections"

  id = Column(Integer, primary_key=True)
  object_id = Column(Integer)
  class_name = Column(String)
  confidence = Column(Float)
  x1 = Column(Integer)
  y1 = Column(Integer)
  x2 = Column(Integer)
  y2 = Column(Integer)
  frame_number = Column(Integer)
  time_stamp = Column(DateTime)
  source = Column(String)


class Detection_repo():
  def __init__(self, detection:Detection):
    global engine
    self.detection = detection
    self.Session = sessionmaker(bind=engine)
    self.session = self.Session()

  def save(self): 
    self.session.add(self.detection)
    self.session.commit()


# set up engine after the Detection class
engine = create_engine("sqlite:///detections.db")
Base.metadata.create_all(engine)