class Protein:
    def __init__(self, acronym: str):
        self.acronym = acronym.lower()

class Molecule:
    def __init__(self, id: int, smile: str, is_binded: bool = None, binding_affinity: float = None):
        self.id = id
        self.smile = smile
        self.is_binded = is_binded
        self.binding_affinity = binding_affinity