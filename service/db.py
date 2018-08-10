
class PersonDatabase:
    """Database holding registered persons"""

   def __init__(self, database):
        self.database = database
        FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        load_weights_from_FaceNet(FRmodel)
        self.model = FRmodel 
