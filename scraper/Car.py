class Car:
    def __init__(self, price, model, model_year, first_reg, km, color, gear, hjuldrift, drivstoff, effekt, sylindervolum, finn_kode):
        self.price = price
        self.model = model

        self.model_year = model_year
        self.first_reg = first_reg
        self.km = km
        self.color = color
        self.gear = gear
        self.hjuldrift = hjuldrift
        self.drivstoff = drivstoff
        self.effekt = effekt
        self.sylindervolum = sylindervolum
        self.finn_kode = finn_kode

    def to_dict(self):
        return {
            'price': self.price,
            'model': self.model,
            'model_year': self.model_year,
            'first_reg': self.first_reg,
            'km': self.km,
            'color': self.color,
            'gear': self.gear,
            'trans': self.hjuldrift,
            'fuel_type': self.drivstoff,
            'power': self.effekt,
            'cylinder': self.sylindervolum,
            'finn_code': self.finn_kode
        }

    def __iter__(self):
        return iter([self.price, self.model, self.model_year, self.first_reg, self.km, self.color, self.gear, self.hjuldrift, self.drivstoff, self.effekt, self.sylindervolum, self.finn_kode])

