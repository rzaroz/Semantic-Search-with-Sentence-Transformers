# Dependencies
import numpy as np
from sentence_transformers import SentenceTransformer

# Some random data
DOCUMENTS = [
    # Technology and Programming
    "Artificial intelligence is transforming modern industries by automating complex tasks and improving decision-making processes.",
    "Python is widely used in scientific computing, data analysis, machine learning, and backend software development.",
    "Cloud computing allows organizations to access scalable computing resources without maintaining physical infrastructure.",
    "Cybersecurity focuses on protecting computer systems, networks, and sensitive information from unauthorized access.",
    "Blockchain technology maintains distributed records through cryptographic techniques and decentralized consensus mechanisms.",
    "Operating systems manage hardware resources and provide an environment for applications to execute efficiently.",
    "Database indexing improves query performance by reducing the amount of data that must be scanned.",
    "Distributed systems coordinate multiple computers to perform tasks that would otherwise require a single powerful machine.",
    "Containerization allows applications to run consistently across different environments using isolated execution units.",
    "Version control systems help developers track changes, collaborate on projects, and restore previous versions of their code.",

    # Science and Physics
    "Gravity is a fundamental interaction that attracts objects with mass toward one another throughout the universe.",
    "Light travels through a vacuum at approximately three hundred thousand kilometers per second.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The periodic table organizes chemical elements according to their atomic numbers and recurring chemical properties.",
    "Photosynthesis allows plants to convert sunlight, water, and carbon dioxide into chemical energy.",
    "DNA contains genetic instructions that determine the development and functioning of living organisms.",
    "Black holes are astronomical objects whose gravitational fields are so strong that even light cannot escape.",
    "Thermodynamics studies the relationships between heat, temperature, energy, and mechanical work.",
    "Electromagnetic waves can travel through empty space without requiring a physical medium.",
    "The Earth's magnetic field helps protect the planet from charged particles emitted by the Sun.",

    # Geography and Environment
    "The Amazon rainforest contains an extraordinary diversity of plant and animal species.",
    "Mount Everest is the highest mountain above sea level and is located in the Himalayan mountain range.",
    "The Sahara Desert covers a large portion of North Africa and experiences extremely low annual precipitation.",
    "Ocean currents distribute heat around the planet and influence regional weather patterns.",
    "Deforestation reduces natural habitats and contributes to biodiversity loss and climate change.",
    "Renewable energy sources include solar power, wind energy, hydropower, and geothermal energy.",
    "Earthquakes occur when accumulated geological stress is suddenly released along faults in the Earth's crust.",
    "Coral reefs provide essential habitats for marine organisms and support complex underwater ecosystems.",
    "Glaciers store significant quantities of freshwater and gradually reshape landscapes through erosion.",
    "Air pollution can negatively affect respiratory health and contribute to environmental degradation.",

    # History and Civilization
    "Ancient Egyptian civilization developed along the Nile River and became famous for its monumental architecture.",
    "The Roman Empire established extensive transportation networks that connected territories across Europe and the Mediterranean.",
    "The Industrial Revolution transformed manufacturing through mechanization, factories, and new energy sources.",
    "The invention of the printing press significantly increased the availability of written information in Europe.",
    "Ancient Mesopotamia is considered one of the earliest centers of urban civilization and organized agriculture.",
    "The Silk Road connected commercial centers across Asia, the Middle East, and Europe for centuries.",
    "The Renaissance encouraged developments in art, science, architecture, and philosophical thinking.",
    "The construction of the Great Wall of China involved multiple dynasties and generations of workers.",
    "The development of maritime navigation expanded international trade and geographical exploration.",
    "The discovery of ancient archaeological sites provides valuable information about forgotten civilizations.",

    # Health and Medicine
    "Regular physical activity improves cardiovascular health and supports the maintenance of healthy body weight.",
    "The human immune system identifies and responds to pathogens that may cause infectious diseases.",
    "Vaccines train the immune system to recognize specific microorganisms and reduce the risk of severe illness.",
    "Adequate sleep supports memory consolidation, hormonal regulation, and the recovery of physical tissues.",
    "Balanced nutrition provides essential vitamins, minerals, proteins, fats, and carbohydrates.",
    "Antibiotics are medications used to treat bacterial infections and are ineffective against viral diseases.",
    "Blood circulation transports oxygen and nutrients to tissues while removing metabolic waste products.",
    "The human brain contains billions of neurons that communicate through electrical and chemical signals.",
    "Preventive healthcare includes regular examinations, vaccinations, and lifestyle modifications.",
    "Chronic stress can influence hormonal balance, cardiovascular function, and psychological well-being.",

    # Economics and Finance
    "Inflation represents a sustained increase in the general price level of goods and services within an economy.",
    "Central banks use monetary policy instruments to influence interest rates and manage economic stability.",
    "Stock markets allow companies to raise capital while providing investors with opportunities to purchase ownership shares.",
    "Diversification reduces investment risk by distributing capital across different assets and financial instruments.",
    "Exchange rates determine how much one currency is worth when converted into another currency.",
    "Compound interest allows investments to grow as accumulated interest generates additional returns over time.",
    "Supply and demand interact to determine market prices and the quantity of goods exchanged.",
    "Government taxation provides revenue for public services, infrastructure, education, and healthcare.",
    "Recessions are periods of declining economic activity that may involve lower production and higher unemployment.",
    "International trade enables countries to exchange products and services based on economic specialization.",

    # Mathematics and Education
    "Linear algebra studies vectors, matrices, and linear transformations used in many scientific applications.",
    "Calculus provides mathematical tools for analyzing continuous change and calculating accumulated quantities.",
    "Probability theory measures uncertainty and helps estimate the likelihood of future events.",
    "Statistics involves collecting, analyzing, interpreting, and presenting numerical information.",
    "A prime number is a natural number greater than one that has exactly two positive divisors.",
    "Machine learning algorithms identify patterns in datasets and use them to make predictions.",
    "Scientific research requires systematic observation, hypothesis formation, experimentation, and evidence evaluation.",
    "Critical thinking involves evaluating information, identifying assumptions, and examining logical arguments.",
    "Educational technology provides digital tools that support interactive learning and remote collaboration.",
    "Learning a foreign language improves communication skills and provides access to different cultures.",

    # Sports and Entertainment
    "Football is one of the most popular sports in the world and is played by two teams of eleven players.",
    "Basketball requires players to score points by throwing a ball through the opposing team's elevated hoop.",
    "Swimming improves cardiovascular endurance while engaging multiple muscle groups throughout the body.",
    "Chess is a strategic board game in which players attempt to checkmate their opponent's king.",
    "The Olympic Games bring together athletes from different countries to compete in international sporting events.",
    "Video games have evolved into complex interactive experiences featuring realistic graphics and sophisticated storytelling.",
    "Role-playing games allow players to control characters and influence fictional worlds through their decisions.",
    "Photography combines technical knowledge and artistic composition to capture meaningful visual moments.",
    "Classical music has influenced numerous musical traditions and continues to be performed worldwide.",
    "Cinema combines visual storytelling, sound design, acting, and cinematography to create immersive experiences.",

    # Agriculture and Food
    "Agriculture provides essential food resources and supports the economic activities of many rural communities.",
    "Crop rotation helps maintain soil fertility and reduces the accumulation of certain agricultural pests.",
    "Irrigation systems distribute water to agricultural fields in regions where natural rainfall is insufficient.",
    "Honeybees contribute to agricultural productivity by pollinating many flowering plants and food crops.",
    "Fermentation is a biological process used to produce foods such as yogurt, cheese, bread, and pickled vegetables.",
    "Olive oil is widely used in Mediterranean cuisine and contains several beneficial fatty acids.",
    "Rice is a staple food for billions of people and is cultivated in many different climatic regions.",
    "Greenhouse farming allows agricultural producers to control environmental conditions for plant growth.",
    "Food preservation techniques include refrigeration, freezing, drying, canning, and fermentation.",
    "Soil quality depends on mineral composition, organic matter, moisture, and biological activity.",

    # Architecture and Engineering
    "Modern architecture combines structural engineering, functional design, environmental considerations, and aesthetics.",
    "Reinforced concrete combines concrete and steel reinforcement to improve structural performance under different loads.",
    "Bridges are engineered structures designed to provide safe transportation across obstacles such as rivers and valleys.",
    "Earthquake-resistant buildings use structural systems that help absorb and distribute seismic forces.",
    "Urban planning coordinates transportation, housing, infrastructure, and public spaces within cities.",
    "Electric vehicles use electric motors and rechargeable batteries instead of conventional internal combustion engines.",
    "Robotics combines mechanical engineering, electronics, and computer science to create programmable machines.",
    "Three-dimensional printing produces physical objects by depositing material in successive layers.",
    "Wind turbines convert the kinetic energy of moving air into electrical power.",
    "Modern transportation networks depend on integrated systems of roads, railways, airports, and ports.",
]

# Load model
#  According to hugging face:
# This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
print("Loading model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Model loaded!")

# Embedding whole document
doc_embeddings = model.encode(DOCUMENTS, normalize_embeddings=True)

def search(query: str, top_k: int = 3):
    """
    query: input sentence,
    top: Number of acceptable sentences
    """

    # Searching
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    # Comparison
    scores = doc_embeddings @ q_emb

    # Final result
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(DOCUMENTS[i], float(scores[i])) for i in top_idx]

if __name__ == "__main__":
    print('Ready for extraction; type "exit" , "quit" or "q" to exit:')
    while True:
        query = input("🔎 New sentence: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        for rank, (text, score) in enumerate(search(query), start=1):
            print(f"  {rank}. [{score:.3f}] {text}")
        print()