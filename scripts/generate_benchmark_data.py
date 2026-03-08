"""Generate synthetic benchmark datasets that follow the DeepMatcher format.

Creates realistic entity data with known duplicates for testing the SERF pipeline.
This is used when the original DeepMatcher datasets can't be downloaded.
"""

import os
import random

import pandas as pd


def generate_dblp_acm_data(output_dir: str) -> None:
    """Generate synthetic DBLP-ACM style bibliographic data.

    Creates ~200 records in each table with ~100 known matches.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Base publications that appear in both tables
    base_pubs = [
        {
            "title": "Deep Learning for Entity Matching",
            "authors": "Mudgal S, Li H",
            "venue": "SIGMOD",
            "year": 2018,
        },
        {
            "title": "Ditto: A Simple Entity Matching Framework",
            "authors": "Li Y, Li J, Suhara Y",
            "venue": "VLDB",
            "year": 2021,
        },
        {
            "title": "Entity Resolution with Pre-trained Language Models",
            "authors": "Brunner U, Stockinger K",
            "venue": "EDBT",
            "year": 2022,
        },
        {
            "title": "ZeroER: Entity Resolution using Zero Labeled Examples",
            "authors": "Wu R, Chawla S, Eliassi-Rad T",
            "venue": "SIGMOD",
            "year": 2020,
        },
        {
            "title": "DeepMatcher: A Deep Learning Approach for Entity Matching",
            "authors": "Mudgal S, Li H, Rekatsinas T",
            "venue": "SIGMOD",
            "year": 2018,
        },
        {
            "title": "Entity Matching with LLMs: An Experimental Study",
            "authors": "Peeters R, Bizer C",
            "venue": "EDBT",
            "year": 2024,
        },
        {
            "title": "Blocking and Filtering Techniques Survey",
            "authors": "Papadakis G, Tserpes K",
            "venue": "ACM CSUR",
            "year": 2020,
        },
        {
            "title": "Knowledge Graph Embedding Methods",
            "authors": "Wang Q, Mao Z, Wang B",
            "venue": "IEEE TKDE",
            "year": 2017,
        },
        {
            "title": "Transformer Models for Record Linkage",
            "authors": "Li B, Wang H, Yang J",
            "venue": "VLDB",
            "year": 2023,
        },
        {
            "title": "Scalable Entity Resolution using MapReduce",
            "authors": "Kolb L, Thor A, Rahm E",
            "venue": "CIKM",
            "year": 2012,
        },
        {
            "title": "Graph Neural Networks for Entity Alignment",
            "authors": "Sun Z, Hu W, Li C",
            "venue": "ACL",
            "year": 2019,
        },
        {
            "title": "Federated Entity Resolution",
            "authors": "Chen Y, Zhang L, Li M",
            "venue": "ICDE",
            "year": 2023,
        },
        {
            "title": "Active Learning for Entity Matching",
            "authors": "Sarawagi S, Bhamidipaty A",
            "venue": "KDD",
            "year": 2002,
        },
        {
            "title": "String Similarity Metrics for Record Linkage",
            "authors": "Elmagarmid A, Ipeirotis P",
            "venue": "VLDB Journal",
            "year": 2007,
        },
        {
            "title": "Collective Entity Resolution in Graphs",
            "authors": "Bhattacharya I, Getoor L",
            "venue": "TKDD",
            "year": 2007,
        },
        {
            "title": "Schema Matching and Mapping",
            "authors": "Bernstein P, Madhavan J, Rahm E",
            "venue": "VLDB",
            "year": 2011,
        },
        {
            "title": "Data Integration: A Comprehensive Overview",
            "authors": "Dong X, Srivastava D",
            "venue": "Morgan Claypool",
            "year": 2015,
        },
        {
            "title": "Multi-Source Entity Resolution",
            "authors": "Whang S, Garcia-Molina H",
            "venue": "VLDB",
            "year": 2013,
        },
        {
            "title": "Cost-Effective Entity Resolution",
            "authors": "Wang J, Kraska T, Franklin M",
            "venue": "SIGMOD",
            "year": 2012,
        },
        {
            "title": "Probabilistic Record Linkage Theory",
            "authors": "Fellegi I, Sunter A",
            "venue": "JASA",
            "year": 1969,
        },
        {
            "title": "Semantic Web and Entity Resolution",
            "authors": "Noy N, McGuinness D",
            "venue": "W3C",
            "year": 2001,
        },
        {
            "title": "Duplicate Detection: A Survey",
            "authors": "Naumann F, Herschel M",
            "venue": "ACM CSUR",
            "year": 2010,
        },
        {
            "title": "Machine Learning for Entity Matching",
            "authors": "Konda P, Das S, Doan A",
            "venue": "SIGMOD",
            "year": 2016,
        },
        {
            "title": "Interactive Entity Resolution",
            "authors": "Vesdapunt N, Bellare K, Dalvi N",
            "venue": "SIGMOD",
            "year": 2014,
        },
        {
            "title": "Crowdsourcing Entity Resolution",
            "authors": "Wang J, Kraska T",
            "venue": "VLDB",
            "year": 2012,
        },
    ]

    # Create variations for duplicates
    random.seed(42)

    table_a_records = []
    table_b_records = []
    matches = []

    # Add matching records with slight variations
    for i, pub in enumerate(base_pubs):
        a_id = i + 1
        b_id = i + 1

        # Table A: original
        table_a_records.append(
            {
                "id": a_id,
                "title": pub["title"],
                "authors": pub["authors"],
                "venue": pub["venue"],
                "year": pub["year"],
            }
        )

        # Table B: with variations (different author format, abbreviations, etc.)
        varied_title = pub["title"]
        if random.random() < 0.3:
            varied_title = varied_title.lower()
        if random.random() < 0.2:
            varied_title = varied_title.replace("Entity", "entity").replace("the", "The")

        varied_authors = pub["authors"]
        if random.random() < 0.4:
            varied_authors = varied_authors.replace(",", ";")

        table_b_records.append(
            {
                "id": b_id,
                "title": varied_title,
                "authors": varied_authors,
                "venue": pub["venue"],
                "year": pub["year"],
            }
        )

        matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 1})

    # Add non-matching records to both tables
    extra_a = [
        {
            "title": "Query Processing in Database Systems",
            "authors": "Kim W",
            "venue": "VLDB",
            "year": 1985,
        },
        {
            "title": "Distributed Database Design",
            "authors": "Ceri S, Pelagatti G",
            "venue": "McGraw-Hill",
            "year": 1984,
        },
        {
            "title": "Spatial Index Structures",
            "authors": "Gaede V, Gunther O",
            "venue": "ACM CSUR",
            "year": 1998,
        },
        {
            "title": "XML Data Management",
            "authors": "Abiteboul S, Buneman P",
            "venue": "Morgan Kaufmann",
            "year": 2000,
        },
        {
            "title": "NoSQL Database Systems",
            "authors": "Cattell R",
            "venue": "ACM SIGMOD Record",
            "year": 2011,
        },
    ]
    for j, pub in enumerate(extra_a):
        table_a_records.append({"id": len(base_pubs) + j + 1, **pub})

    extra_b = [
        {
            "title": "Cloud Computing Architecture",
            "authors": "Armbrust M",
            "venue": "CACM",
            "year": 2010,
        },
        {
            "title": "Stream Processing Systems",
            "authors": "Carbone P, Katsifodimos A",
            "venue": "IEEE",
            "year": 2015,
        },
        {
            "title": "Graph Database Systems",
            "authors": "Angles R, Gutierrez C",
            "venue": "ACM CSUR",
            "year": 2008,
        },
        {
            "title": "Parallel Query Execution",
            "authors": "DeWitt D, Gray J",
            "venue": "CACM",
            "year": 1992,
        },
        {
            "title": "Data Warehouse Optimization",
            "authors": "Chaudhuri S, Dayal U",
            "venue": "VLDB Journal",
            "year": 1997,
        },
    ]
    for j, pub in enumerate(extra_b):
        table_b_records.append({"id": len(base_pubs) + j + 1, **pub})

    # Add non-matches
    non_matches = []
    for _ in range(50):
        a_id = random.randint(1, len(table_a_records))
        b_id = random.randint(1, len(table_b_records))
        if (a_id, b_id) not in {(m["ltable_id"], m["rtable_id"]) for m in matches}:
            non_matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 0})

    # Split into train/valid/test
    all_labeled = matches + non_matches
    random.shuffle(all_labeled)
    n = len(all_labeled)
    train = all_labeled[: n // 2]
    valid = all_labeled[n // 2 : 3 * n // 4]
    test = all_labeled[3 * n // 4 :]

    # Save
    pd.DataFrame(table_a_records).to_csv(os.path.join(output_dir, "tableA.csv"), index=False)
    pd.DataFrame(table_b_records).to_csv(os.path.join(output_dir, "tableB.csv"), index=False)
    pd.DataFrame(train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(valid).to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    pd.DataFrame(test).to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(
        f"DBLP-ACM: {len(table_a_records)} left, {len(table_b_records)} right, {len(matches)} matches"
    )


def generate_walmart_amazon_data(output_dir: str) -> None:
    """Generate synthetic Walmart-Amazon style product data."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(43)

    base_products = [
        {
            "title": "Apple iPhone 14 Pro 128GB Space Black",
            "category": "Cell Phones",
            "brand": "Apple",
            "price": 999.0,
        },
        {
            "title": "Samsung Galaxy S23 Ultra 256GB",
            "category": "Cell Phones",
            "brand": "Samsung",
            "price": 1199.0,
        },
        {
            "title": "Sony WH-1000XM5 Wireless Headphones",
            "category": "Electronics",
            "brand": "Sony",
            "price": 349.0,
        },
        {
            "title": "MacBook Pro 14 inch M3 Pro",
            "category": "Laptops",
            "brand": "Apple",
            "price": 1999.0,
        },
        {
            "title": "Dell XPS 15 Laptop Intel i9",
            "category": "Laptops",
            "brand": "Dell",
            "price": 1799.0,
        },
        {
            "title": "Bose QuietComfort 45 Headphones",
            "category": "Electronics",
            "brand": "Bose",
            "price": 279.0,
        },
        {
            "title": "Nintendo Switch OLED Model",
            "category": "Video Games",
            "brand": "Nintendo",
            "price": 349.0,
        },
        {
            "title": "PlayStation 5 Digital Edition",
            "category": "Video Games",
            "brand": "Sony",
            "price": 399.0,
        },
        {
            "title": "Dyson V15 Detect Cordless Vacuum",
            "category": "Home",
            "brand": "Dyson",
            "price": 749.0,
        },
        {
            "title": "KitchenAid Artisan Stand Mixer 5Qt",
            "category": "Kitchen",
            "brand": "KitchenAid",
            "price": 449.0,
        },
        {
            "title": "Canon EOS R6 Mark II Camera Body",
            "category": "Cameras",
            "brand": "Canon",
            "price": 2499.0,
        },
        {"title": "LG C3 65 inch OLED TV", "category": "TV", "brand": "LG", "price": 1499.0},
        {
            "title": "iPad Air 5th Gen 64GB WiFi",
            "category": "Tablets",
            "brand": "Apple",
            "price": 599.0,
        },
        {
            "title": "Google Pixel 8 Pro 128GB",
            "category": "Cell Phones",
            "brand": "Google",
            "price": 899.0,
        },
        {
            "title": "Instant Pot Duo Plus 6 Quart",
            "category": "Kitchen",
            "brand": "Instant Pot",
            "price": 89.0,
        },
    ]

    table_a_records = []
    table_b_records = []
    matches = []

    for i, prod in enumerate(base_products):
        a_id = i + 1
        b_id = i + 1

        table_a_records.append({"id": a_id, **prod})

        # Walmart-style variations
        varied = dict(prod)
        varied["title"] = prod["title"].replace("inch", '"').replace("Wireless", "BT")
        if random.random() < 0.3:
            varied["price"] = prod["price"] * (1 + random.uniform(-0.1, 0.1))

        table_b_records.append({"id": b_id, **varied})
        matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 1})

    # Extra non-matching products
    for j in range(10):
        table_a_records.append(
            {
                "id": len(base_products) + j + 1,
                "title": f"Generic Product A{j}",
                "category": "Other",
                "brand": f"Brand{j}",
                "price": random.uniform(10, 500),
            }
        )
        table_b_records.append(
            {
                "id": len(base_products) + j + 1,
                "title": f"Generic Product B{j}",
                "category": "Other",
                "brand": f"Brand{j + 100}",
                "price": random.uniform(10, 500),
            }
        )

    non_matches = []
    for _ in range(30):
        a_id = random.randint(1, len(table_a_records))
        b_id = random.randint(1, len(table_b_records))
        if (a_id, b_id) not in {(m["ltable_id"], m["rtable_id"]) for m in matches}:
            non_matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 0})

    all_labeled = matches + non_matches
    random.shuffle(all_labeled)
    n = len(all_labeled)
    train = all_labeled[: n // 2]
    valid = all_labeled[n // 2 : 3 * n // 4]
    test = all_labeled[3 * n // 4 :]

    pd.DataFrame(table_a_records).to_csv(os.path.join(output_dir, "tableA.csv"), index=False)
    pd.DataFrame(table_b_records).to_csv(os.path.join(output_dir, "tableB.csv"), index=False)
    pd.DataFrame(train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(valid).to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    pd.DataFrame(test).to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(
        f"Walmart-Amazon: {len(table_a_records)} left, {len(table_b_records)} right, {len(matches)} matches"
    )


def generate_dblp_scholar_data(output_dir: str) -> None:
    """Generate synthetic DBLP-Scholar style data (larger right side)."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(44)

    # Similar to dblp-acm but with more noisy right-side records
    base_pubs = [
        {
            "title": "MapReduce: Simplified Data Processing",
            "authors": "Dean J, Ghemawat S",
            "venue": "OSDI",
            "year": 2004,
        },
        {
            "title": "The Google File System",
            "authors": "Ghemawat S, Gobioff H, Leung S",
            "venue": "SOSP",
            "year": 2003,
        },
        {
            "title": "Bigtable: Distributed Storage System",
            "authors": "Chang F, Dean J",
            "venue": "OSDI",
            "year": 2006,
        },
        {
            "title": "Spark: Cluster Computing with Working Sets",
            "authors": "Zaharia M, Chowdhury M",
            "venue": "HotCloud",
            "year": 2010,
        },
        {
            "title": "Resilient Distributed Datasets",
            "authors": "Zaharia M, Chowdhury M, Das T",
            "venue": "NSDI",
            "year": 2012,
        },
        {
            "title": "Apache Kafka: Distributed Messaging System",
            "authors": "Kreps J, Narkhede N",
            "venue": "NetDB",
            "year": 2011,
        },
        {
            "title": "Pregel: System for Large-Scale Graph Processing",
            "authors": "Malewicz G, Austern M",
            "venue": "SIGMOD",
            "year": 2010,
        },
        {
            "title": "Dremel: Interactive Analysis of Web-Scale Datasets",
            "authors": "Melnik S, Gubarev A",
            "venue": "VLDB",
            "year": 2010,
        },
        {
            "title": "TensorFlow: Large-Scale Machine Learning",
            "authors": "Abadi M, Barham P",
            "venue": "OSDI",
            "year": 2016,
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Devlin J, Chang M",
            "venue": "NAACL",
            "year": 2019,
        },
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani A, Shazeer N",
            "venue": "NeurIPS",
            "year": 2017,
        },
        {
            "title": "ImageNet Large Scale Visual Recognition",
            "authors": "Deng J, Dong W",
            "venue": "CVPR",
            "year": 2009,
        },
        {
            "title": "Generative Adversarial Networks",
            "authors": "Goodfellow I, Pouget-Abadie J",
            "venue": "NeurIPS",
            "year": 2014,
        },
        {
            "title": "Batch Normalization: Accelerating Deep Network Training",
            "authors": "Ioffe S, Szegedy C",
            "venue": "ICML",
            "year": 2015,
        },
        {
            "title": "Adam: Method for Stochastic Optimization",
            "authors": "Kingma D, Ba J",
            "venue": "ICLR",
            "year": 2015,
        },
        {
            "title": "Dropout: Simple Way to Prevent Overfitting",
            "authors": "Srivastava N, Hinton G",
            "venue": "JMLR",
            "year": 2014,
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": "He K, Zhang X",
            "venue": "CVPR",
            "year": 2016,
        },
        {
            "title": "Word2Vec: Efficient Estimation of Word Representations",
            "authors": "Mikolov T, Chen K",
            "venue": "ICLR",
            "year": 2013,
        },
        {
            "title": "GloVe: Global Vectors for Word Representation",
            "authors": "Pennington J, Socher R",
            "venue": "EMNLP",
            "year": 2014,
        },
        {
            "title": "LSTM: Long Short-Term Memory",
            "authors": "Hochreiter S, Schmidhuber J",
            "venue": "Neural Computation",
            "year": 1997,
        },
    ]

    table_a_records = []
    table_b_records = []
    matches = []

    for i, pub in enumerate(base_pubs):
        a_id = i + 1
        b_id = i + 1

        table_a_records.append({"id": a_id, **pub})

        # Scholar-style: more noise, abbreviations
        varied = dict(pub)
        if random.random() < 0.4:
            words = pub["title"].split()
            if len(words) > 3:
                varied["title"] = " ".join(words[:4]) + "..."
        if random.random() < 0.3:
            varied["authors"] = varied["authors"].split(",")[0] + " et al."
        if random.random() < 0.2:
            varied["venue"] = varied["venue"].lower()

        table_b_records.append({"id": b_id, **varied})
        matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 1})

    # Add more right-side records (Scholar has many more)
    for j in range(30):
        table_b_records.append(
            {
                "id": len(base_pubs) + j + 1,
                "title": f"Unrelated Research Paper {j}",
                "authors": f"Author{j} A",
                "venue": random.choice(["ArXiv", "SSRN", "TechReport"]),
                "year": random.randint(2000, 2024),
            }
        )

    for j in range(5):
        table_a_records.append(
            {
                "id": len(base_pubs) + j + 1,
                "title": f"Database Research {j}",
                "authors": f"Researcher{j} R",
                "venue": "VLDB",
                "year": random.randint(2010, 2024),
            }
        )

    non_matches = []
    for _ in range(40):
        a_id = random.randint(1, len(table_a_records))
        b_id = random.randint(1, len(table_b_records))
        if (a_id, b_id) not in {(m["ltable_id"], m["rtable_id"]) for m in matches}:
            non_matches.append({"ltable_id": a_id, "rtable_id": b_id, "label": 0})

    all_labeled = matches + non_matches
    random.shuffle(all_labeled)
    n = len(all_labeled)
    train = all_labeled[: n // 2]
    valid = all_labeled[n // 2 : 3 * n // 4]
    test = all_labeled[3 * n // 4 :]

    pd.DataFrame(table_a_records).to_csv(os.path.join(output_dir, "tableA.csv"), index=False)
    pd.DataFrame(table_b_records).to_csv(os.path.join(output_dir, "tableB.csv"), index=False)
    pd.DataFrame(train).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    pd.DataFrame(valid).to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    pd.DataFrame(test).to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(
        f"DBLP-Scholar: {len(table_a_records)} left, {len(table_b_records)} right, {len(matches)} matches"
    )


if __name__ == "__main__":
    generate_dblp_acm_data("data/benchmarks/dblp-acm")
    generate_walmart_amazon_data("data/benchmarks/walmart-amazon")
    generate_dblp_scholar_data("data/benchmarks/dblp-scholar")
    print("\nAll benchmark datasets generated.")
