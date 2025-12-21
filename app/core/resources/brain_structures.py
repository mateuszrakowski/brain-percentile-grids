from pydantic import BaseModel


class CerebralCortex(BaseModel):
    cerebral_cortex_left: str = "Kora_mózgu_lewa"
    cerebral_cortex_right: str = "Kora_mózgu_prawa"


class CerebralCerebellumCortex(CerebralCortex):
    cerebellum_grey_matter_left: str = "Istota_szara_móżdżku_lewa"
    cerebellum_grey_matter_right: str = "Istota_szara_móżdżku_prawa"


class SubcorticalGreyMatter(BaseModel):
    thalamus_left: str = "Wzgórze_lewe"
    thalamus_right: str = "Wzgórze_prawe"
    caudate_nucleus_left: str = "Jądro_ogoniaste_lewe"
    caudate_nucleus_right: str = "Jądro_ogoniaste_prawe"
    paleostriatum_left: str = "Gałka_blada_lewa"
    paleostriatum_right: str = "Gałka_blada_prawa"
    amygdala_left: str = "Ciało_migdałowate_lewe"
    amygdala_right: str = "Ciało_migdałowate_prawe"
    nucleus_accumbens_left: str = "Jądro_półleżące_lewe"
    nucleus_accumbens_right: str = "Jądro_półleżące_prawe"
    putamen_left: str = "Skorupa_lewa"
    putamen_right: str = "Skorupa_prawa"
    hippocampus_left: str = "Hipokamp_lewy"
    hippocampus_right: str = "Hipokamp_prawy"


class WhiteMatterCerebral(BaseModel):
    white_matter_cerebral_left: str = "Istota_biała_lewa"
    white_matter_cerebral_right: str = "Istota_biała_prawa"


class WhiteMatterTotal(WhiteMatterCerebral):
    amygdala_left: str = "Ciało_migdałowate_lewe"
    amygdala_right: str = "Ciało_migdałowate_prawe"
    brainstem: str = "Pień_mózgu"
    white_matter_cerebellum_left: str = "Istota_biała_móżdżku_lewa"
    white_matter_cerebellum_right: str = "Istota_biała_móżdżku_prawa"
    midbrain_left: str = "Międzymózgowie_lewe"
    midbrain_right: str = "Międzymózgowie_prawe"
    optic_chiasm: str = "Skrzyżowanie_wzrokowe"


class NeuralStructuresTotal(
    CerebralCerebellumCortex, SubcorticalGreyMatter, WhiteMatterTotal
):
    pass


class VentricularSupratentorialSystem(BaseModel):
    ventricular_system_left: str = "Układ_komorowy_mózgu_lewy"
    ventricular_system_right: str = "Układ_komorowy_mózgu_prawy"
    third_ventricle: str = "Komora_trzecia"


class CerebrospinalFluidTotal(VentricularSupratentorialSystem):
    cerebrospinal_fluid: str = "Płyn_mózgowo_rdzeniowy"
    fourth_ventricle: str = "Komora_czwarta"


class TotalStructuresVolume(NeuralStructuresTotal, CerebrospinalFluidTotal):
    pass
