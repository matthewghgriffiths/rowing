
AbsoluteZeroC = -273.15
SpecificGasConstantDryAir = 287.0531
SpecificGasConstantWaterVapour = 461.4964


class AirDensity:
    def __init__(self, Pressure_hPa, Temp_C, RelativeHumidity):
        self.Pressure_hPa = Pressure_hPa
        self.Temp_C = Temp_C
        self.RelativeHumidity = RelativeHumidity

    @property
    def Pressure_Pa(self):
        return self.Pressure_hPa*100

    @property
    def Temp_K(self):
        return self.Temp_C+273.15

    @property
    def Es_Pa(self):
        Temp_C = self.Temp_C
        Eso = 6.1078
        c0 = 0.99999683
        c1 = -0.90826951*10**-2
        c2 = 0.78736169*10**-4
        c3 = -0.61117958*10**-6
        c4 = 0.43884187*10**-8
        c5 = -0.29883885*10**-10
        c6 = 0.21874425*10**-12
        c7 = -0.17892321*10**-14
        c8 = 0.11112018*10**-16
        c9 = -0.30994571*10**-19
        p = c0+Temp_C*(c1+Temp_C*(c2+Temp_C*(c3+Temp_C*(c4+Temp_C *
                       (c5+Temp_C*(c6+Temp_C*(c7+Temp_C*(c8+Temp_C*(c9)))))))))
        Es_hPa = Eso/(p**8)
        return Es_hPa*100

    @property
    def PartialPressureWaterVapour_Pa(self):
        return self.Es_Pa * self.RelativeHumidity

    @property
    def PartialPressureDryAir_Pa(self):
        return self.Pressure_Pa - self.PartialPressureWaterVapour_Pa

    @property
    def DensityHumidAir(self):
        return self.PartialPressureDryAir_Pa / (
            SpecificGasConstantDryAir*self.Temp_K
        ) + self.PartialPressureWaterVapour_Pa / (
            SpecificGasConstantWaterVapour*self.Temp_K
        )


def air_density(Pressure_hPa, Temp_C, RelativeHumidity):
    return AirDensity(Pressure_hPa, Temp_C, RelativeHumidity).DensityHumidAir
