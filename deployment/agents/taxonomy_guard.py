"""Agent 3: Biological Taxonomy & Host-Pathogen Guard Agent.

Enforces fundamental botanical and phytopathological constraints based on the
comprehensive PlantVillage taxonomy (covering all 14 crop families and 38 disease categories):
- Verifies that a pathogen biologically infects the host plant family.
  (e.g., Alternaria solani strictly infects Solanaceae, NOT Rosaceae or Apocynaceae).
- Intercepts biologically impossible cross-species misclassifications:
  - Fungal apple scab (Venturia inaequalis) on Solanaceae (Tomato/Potato)
  - Citrus greening (Candidatus Liberibacter) on Grasses (Corn)
  - Insect foliar galls (Pauropsylla depressa) on Solanaceae
- Categorizes symptom morphologies:
  - Foliar Galls (zoo-cecidia from insects/psyllids)
  - Fungal Necrosis & Target Rings (Alternaria, Septoria, Botryosphaeria)
  - Bacterial Spots & Blights (Xanthomonas)
  - Viral Mosaics & Curls (Begomovirus, Tobamovirus)
  - Rusts (Puccinia, Gymnosporangium)
  - Powdery & Velvety Molds (Passalora, Podosphaera)
"""
from typing import Dict, Any, Tuple, List


# Complete PlantVillage + Botanical Host Family Taxonomy (14 Crops + Wild/Non-target Flora)
PLANT_FAMILIES = {
    # Solanaceae (Nightshades)
    "tomato": {"scientific": "Solanum lycopersicum", "family": "Solanaceae", "type": "Herbaceous crop"},
    "potato": {"scientific": "Solanum tuberosum", "family": "Solanaceae", "type": "Tuber crop"},
    "pepper": {"scientific": "Capsicum annuum", "family": "Solanaceae", "type": "Shrub/Herbaceous crop"},
    "bell pepper": {"scientific": "Capsicum annuum", "family": "Solanaceae", "type": "Shrub/Herbaceous crop"},

    # Rosaceae (Rose family)
    "apple": {"scientific": "Malus domestica", "family": "Rosaceae", "type": "Deciduous fruit tree"},
    "cherry": {"scientific": "Prunus avium", "family": "Rosaceae", "type": "Deciduous stone fruit tree"},
    "peach": {"scientific": "Prunus persica", "family": "Rosaceae", "type": "Deciduous stone fruit tree"},
    "strawberry": {"scientific": "Fragaria ananassa", "family": "Rosaceae", "type": "Perennial berry"},
    "raspberry": {"scientific": "Rubus idaeus", "family": "Rosaceae", "type": "Perennial cane fruit"},
    "rose": {"scientific": "Rosa spp.", "family": "Rosaceae", "type": "Ornamental shrub"},

    # Poaceae (Grasses)
    "corn": {"scientific": "Zea mays", "family": "Poaceae", "type": "Cereal grass crop"},
    "maize": {"scientific": "Zea mays", "family": "Poaceae", "type": "Cereal grass crop"},

    # Vitaceae (Grapes)
    "grape": {"scientific": "Vitis vinifera", "family": "Vitaceae", "type": "Woody vine"},

    # Rutaceae (Citrus)
    "orange": {"scientific": "Citrus sinensis", "family": "Rutaceae", "type": "Citrus fruit tree"},
    "citrus": {"scientific": "Citrus spp.", "family": "Rutaceae", "type": "Citrus fruit tree"},

    # Fabaceae (Legumes)
    "soybean": {"scientific": "Glycine max", "family": "Fabaceae", "type": "Legume grain crop"},

    # Cucurbitaceae (Gourds/Melons)
    "squash": {"scientific": "Cucurbita pepo", "family": "Cucurbitaceae", "type": "Vine crop"},

    # Ericaceae (Heaths)
    "blueberry": {"scientific": "Vaccinium corymbosum", "family": "Ericaceae", "type": "Shrub berry"},

    # Apocynaceae (Wild / Non-target Trees)
    "alstonia scholaris": {"scientific": "Alstonia scholaris", "family": "Apocynaceae", "type": "Evergreen tree"},
    "devil tree": {"scientific": "Alstonia scholaris", "family": "Apocynaceae", "type": "Evergreen tree"},
    "saptaparni": {"scientific": "Alstonia scholaris", "family": "Apocynaceae", "type": "Evergreen tree"},
}

# Pathogen Biological Host Range & Phytopathological Signatures
PATHOGEN_HOST_MATRIX = {
    # Solanaceae Pathogens
    "early_blight": {
        "pathogen": "Alternaria solani",
        "kingdom": "Fungi",
        "host_families": ["Solanaceae"],
        "symptom_type": "FUNGAL_NECROSIS",
        "description": "Concentric necrotic rings (bullseye target spots) with yellow chlorotic halos.",
    },
    "late_blight": {
        "pathogen": "Phytophthora infestans",
        "kingdom": "Oomycota",
        "host_families": ["Solanaceae"],
        "symptom_type": "WATER_SOAKED_BLIGHT",
        "description": "Dark water-soaked lesions with white sporulation on undersides in cool, moist conditions.",
    },
    "bacterial_spot": {
        "pathogen": "Xanthomonas campestris / arboricola",
        "kingdom": "Bacteria",
        "host_families": ["Solanaceae", "Rosaceae"],
        "symptom_type": "BACTERIAL_SPOT",
        "description": "Small angular dark spots often with yellow halos or greasy margins.",
    },
    "leaf_mold": {
        "pathogen": "Passalora fulva (Cladosporium)",
        "kingdom": "Fungi",
        "host_families": ["Solanaceae"],
        "symptom_type": "FUNGAL_MOLD",
        "description": "Pale greenish-yellow chlorotic spots on adaxial leaf with olive-brown velvety mold on abaxial.",
    },
    "septoria_leaf_spot": {
        "pathogen": "Septoria lycopersici",
        "kingdom": "Fungi",
        "host_families": ["Solanaceae"],
        "symptom_type": "FUNGAL_SPOT",
        "description": "Small circular spots with dark brown margins and grey/tan centers containing black pycnidia.",
    },
    "target_spot": {
        "pathogen": "Corynespora cassiicola",
        "kingdom": "Fungi",
        "host_families": ["Solanaceae", "Cucurbitaceae"],
        "symptom_type": "FUNGAL_NECROSIS",
        "description": "Brown circular lesions with concentric rings and chlorotic margins on foliage.",
    },
    "spider_mites": {
        "pathogen": "Tetranychus urticae (Two-spotted spider mite)",
        "kingdom": "Animalia (Arachnida)",
        "host_families": ["Solanaceae", "Rosaceae", "Fabaceae", "Cucurbitaceae"],
        "symptom_type": "ARACHNID_STIPPLING",
        "description": "Fine yellow stippling/speckling with visible silky webbing on leaf undersides.",
    },
    "yellow_leaf_curl": {
        "pathogen": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "kingdom": "Virus (Geminiviridae)",
        "host_families": ["Solanaceae"],
        "symptom_type": "VIRAL_CURL",
        "description": "Severe upward leaf curling, stunted leaf size, and interveinal chlorosis transmitted by whiteflies.",
    },
    "mosaic_virus": {
        "pathogen": "Tomato Mosaic Virus (ToMV) / Tobacco Mosaic Virus",
        "kingdom": "Virus (Virgaviridae)",
        "host_families": ["Solanaceae"],
        "symptom_type": "VIRAL_MOSAIC",
        "description": "Light and dark green mottling, blistering, and distortion/filiform deformation of leaves.",
    },

    # Rosaceae Pathogens (Apple, Cherry, Peach, Strawberry, Rose)
    "apple_scab": {
        "pathogen": "Venturia inaequalis",
        "kingdom": "Fungi",
        "host_families": ["Rosaceae"],
        "symptom_type": "FUNGAL_SCAB",
        "description": "Olive-green to velvety dark brown circular scabby lesions on leaves and fruit.",
    },
    "black_rot": {
        "pathogen": "Botryosphaeria obtusa / Guignardia bidwellii",
        "kingdom": "Fungi",
        "host_families": ["Rosaceae", "Vitaceae"],
        "symptom_type": "FUNGAL_ROT",
        "description": "Small reddish-brown circular spots developing tiny black pycnidia bodies; shrivels berries.",
    },
    "cedar_apple_rust": {
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "kingdom": "Fungi",
        "host_families": ["Rosaceae"],
        "symptom_type": "FUNGAL_RUST",
        "description": "Bright yellow-orange spots on upper leaf surface with aecia tubes protruding from underneath.",
    },
    "leaf_scorch": {
        "pathogen": "Diplocarpon earlianum",
        "kingdom": "Fungi",
        "host_families": ["Rosaceae"],
        "symptom_type": "FUNGAL_NECROSIS",
        "description": "Small purplish-red irregular spots coalescing to give burned/scorched appearance.",
    },
    "rose_black_spot": {
        "pathogen": "Diplocarpon rosae",
        "kingdom": "Fungi",
        "host_families": ["Rosaceae"],
        "symptom_type": "FUNGAL_NECROSIS",
        "description": "Circular black spots with feathery fringed margins followed by chlorosis and leaf drop.",
    },

    # Poaceae Pathogens (Corn / Maize)
    "common_rust": {
        "pathogen": "Puccinia sorghi",
        "kingdom": "Fungi",
        "host_families": ["Poaceae"],
        "symptom_type": "FUNGAL_RUST",
        "description": "Cinnamon-brown elongated pustules erupting through both upper and lower leaf surfaces.",
    },
    "northern_leaf_blight": {
        "pathogen": "Exserohilum turcicum",
        "kingdom": "Fungi",
        "host_families": ["Poaceae"],
        "symptom_type": "FUNGAL_BLIGHT",
        "description": "Large elliptical, cigar-shaped grayish-green to tan lesions on corn foliage.",
    },
    "gray_leaf_spot": {
        "pathogen": "Cercospora zeae-maydis",
        "kingdom": "Fungi",
        "host_families": ["Poaceae"],
        "symptom_type": "FUNGAL_SPOT",
        "description": "Rectangular, vein-delimited tan to gray lesions running parallel with corn leaf veins.",
    },

    # Vitaceae Pathogens (Grape)
    "esca": {
        "pathogen": "Phaeomoniella chlamydospora / Fomitiporia",
        "kingdom": "Fungi",
        "host_families": ["Vitaceae"],
        "symptom_type": "FUNGAL_VASCULAR",
        "description": "Interveinal necrotic striping producing a 'tiger-stripe' pattern on grape leaves.",
    },

    # Rutaceae Pathogens (Orange / Citrus)
    "citrus_greening": {
        "pathogen": "Candidatus Liberibacter asiaticus (Huanglongbing)",
        "kingdom": "Bacteria (Fastidious)",
        "host_families": ["Rutaceae"],
        "symptom_type": "BACTERIAL_PHLOEM",
        "description": "Asymmetrical blotchy mottle chlorosis crossing leaf veins, small upright leaves, lopsided bitter fruit.",
    },

    # Cucurbitaceae / Rosaceae Powdery Mildew
    "powdery_mildew": {
        "pathogen": "Podosphaera / Erysiphe spp.",
        "kingdom": "Fungi",
        "host_families": ["Cucurbitaceae", "Rosaceae", "Solanaceae"],
        "symptom_type": "POWDERY_MOLD",
        "description": "White talcum-powder-like fungal patches expanding across leaf surfaces and stems.",
    },

    # Apocynaceae Zoocecidia (Wild / Alstonia)
    "psyllid_leaf_galls": {
        "pathogen": "Pauropsylla depressa",
        "kingdom": "Animalia (Insecta: Psyllidae)",
        "host_families": ["Apocynaceae"],
        "symptom_type": "INSECT_GALL",
        "description": "Prominent raised, blister-like foliar galls caused by gall-inducing psyllid nymphs.",
    },
}


class TaxonomyGuardAgent:
    """Validates biological compatibility between candidate plant hosts and suspected diseases."""

    def __init__(self):
        self.families = PLANT_FAMILIES
        self.matrix = PATHOGEN_HOST_MATRIX

    def resolve_plant_family(self, plant_name: str) -> Dict[str, str]:
        """Resolves common or scientific name to plant family profile."""
        clean = plant_name.lower().strip()
        for key, profile in self.families.items():
            if key in clean or profile["scientific"].lower() in clean:
                return profile
        return {"scientific": "Unknown", "family": "Unknown", "type": "Flora"}

    def validate_pairing(
        self, host_plant: str, disease_name: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Verifies if the disease can biologically infect the specified host plant.

        Returns:
            (is_compatible, explanation, diagnostic_metadata)
        """
        plant_info = self.resolve_plant_family(host_plant)
        host_family = plant_info["family"]

        disease_clean = disease_name.lower().replace(" ", "_").replace("__", "_").replace("-", "_")

        matched_disease_key = None
        for key in self.matrix:
            if key in disease_clean or disease_clean in key:
                matched_disease_key = key
                break

        if not matched_disease_key:
            # Fallback for generic or healthy categories
            if "healthy" in disease_clean:
                return (
                    True,
                    f"Specimen classified as healthy {host_plant} foliage.",
                    {"plant_family": host_family, "symptom_type": "HEALTHY"},
                )
            return (
                True,
                f"No biological incompatibility recorded for '{disease_name}' on {host_plant}.",
                {"plant_family": host_family, "symptom_type": "UNVERIFIED"},
            )

        disease_info = self.matrix[matched_disease_key]
        allowed_families = disease_info["host_families"]

        if host_family == "Unknown":
            return (
                True,
                f"Host family unconfirmed; compatibility with '{disease_info['pathogen']}' is provisional.",
                disease_info,
            )

        if host_family in allowed_families:
            return (
                True,
                f"Biologically compatible: {disease_info['pathogen']} is a confirmed pathogen of {host_family} ({host_plant}).",
                disease_info,
            )

        # Incompatibility detected!
        return (
            False,
            f"BIOLOGICAL INCOMPATIBILITY: {disease_info['pathogen']} strictly targets {', '.join(allowed_families)}, "
            f"and cannot infect {plant_info['scientific']} ({host_family}).",
            disease_info,
        )
