# src/fuzzy_model/model.py

class FuzzyClassifier:
    def __init__(self, rules, membership_params):
        """
        rules: lista de reglas difusas. Cada regla tiene 'if' y 'then'
        membership_params: dict con funciones de membresía por variable y etiqueta
        """
        self.rules = rules
        self.membership_params = membership_params  # Dict[var][label][value]

    def fuzzify(self, var, value):
        """
        Retorna los grados de pertenencia del valor a las etiquetas lingüísticas
        """
        mfs = self.membership_params[var]
        return {label: mfs[label].get(value, 0.0) for label in mfs}

    def predict_single(self, sample):
        """
        Clasifica una muestra usando inferencia difusa Mamdani.
        """
        fuzzy_inputs = {
            var: self.fuzzify(var, sample[var]) for var in self.membership_params
        }

        phishing_strength = 0.0
        legit_strength = 0.0

        for rule in self.rules:
            degrees = [
                fuzzy_inputs[var][label]
                for var, label in rule['if'].items()
            ]
            strength = min(degrees)
            if rule['then'] == 'phishing':
                phishing_strength += strength
            else:
                legit_strength += strength

        return 1 if phishing_strength > legit_strength else -1

    def predict(self, df):
        """
        Clasifica un DataFrame completo.
        """
        return [self.predict_single(row) for _, row in df.iterrows()]
