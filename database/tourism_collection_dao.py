from typing import List, Dict, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    db
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TourismMultimodalDAO:
    """DAO cho du lịch với 2 collections: text search và image search"""

    DATABASE_NAME = "du_lich_db"
    TEXT_COLLECTION_NAME = "tourism_text_search"
    IMAGE_COLLECTION_NAME = "tourism_image_search"

    IMAGE_VECTOR_DIM = 512
    DESCRIPTION_VECTOR_DIM = 768

    def __init__(self, host="localhost", port="19530"):
        """Khởi tạo connection và tạo 2 collections"""
        self.host = host
        self.port = port
        self.connect()
        self.switch_database()
        self.text_collection = self._get_or_create_text_collection()
        self.image_collection = self._get_or_create_image_collection()

    def connect(self):
        """Kết nối tới Milvus server"""
        try:
            try:
                connections.disconnect("default")
            except:
                pass

            logger.info(f"🔌 Connecting to Milvus at {self.host}:{self.port}...")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {e}")
            raise

    def switch_database(self):
        """Chuyển sang database du_lich_db"""
        try:
            databases = db.list_database()
            logger.info(f"📋 Existing databases: {databases}")

            if self.DATABASE_NAME not in databases:
                logger.info(f"🔨 Creating database '{self.DATABASE_NAME}'...")
                db.create_database(self.DATABASE_NAME)
                logger.info(f"✅ Database '{self.DATABASE_NAME}' created")

            db.using_database(self.DATABASE_NAME)
            logger.info(f"✅ Switched to database '{self.DATABASE_NAME}'")

        except Exception as e:
            logger.error(f"❌ Failed to switch database: {e}")
            raise

    def _create_text_schema(self) -> CollectionSchema:
        """Schema cho text search collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(
                name="description_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.DESCRIPTION_VECTOR_DIM
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="Tourism text search collection",
            enable_dynamic_field=True
        )

    def _create_image_schema(self) -> CollectionSchema:
        """Schema cho image search collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="price", dtype=DataType.FLOAT),
            FieldSchema(
                name="image_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.IMAGE_VECTOR_DIM
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="Tourism image search collection",
            enable_dynamic_field=True
        )

    def _get_or_create_text_collection(self) -> Collection:
        """Tạo hoặc load text collection"""
        if utility.has_collection(self.TEXT_COLLECTION_NAME):
            logger.info(f"✅ Text collection '{self.TEXT_COLLECTION_NAME}' exists, loading...")
            collection = Collection(self.TEXT_COLLECTION_NAME)
        else:
            logger.info(f"🔨 Creating text collection '{self.TEXT_COLLECTION_NAME}'")
            schema = self._create_text_schema()
            collection = Collection(name=self.TEXT_COLLECTION_NAME, schema=schema)

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="description_vector", index_params=index_params)
            logger.info("  ✅ Created index for description_vector (COSINE)")

        collection.load()
        logger.info(f"✅ Text collection loaded")
        return collection

    def _get_or_create_image_collection(self) -> Collection:
        """Tạo hoặc load image collection"""
        if utility.has_collection(self.IMAGE_COLLECTION_NAME):
            logger.info(f"✅ Image collection '{self.IMAGE_COLLECTION_NAME}' exists, loading...")
            collection = Collection(self.IMAGE_COLLECTION_NAME)
        else:
            logger.info(f"🔨 Creating image collection '{self.IMAGE_COLLECTION_NAME}'")
            schema = self._create_image_schema()
            collection = Collection(name=self.IMAGE_COLLECTION_NAME, schema=schema)

            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="image_vector", index_params=index_params)
            logger.info("  ✅ Created index for image_vector (L2)")

        collection.load()
        logger.info(f"✅ Image collection loaded")
        return collection

    def insert_data(self, data: List[Dict]) -> Dict[str, List[int]]:
        """
        Chèn dữ liệu vào CẢ 2 collections

        Args:
            data: List các dict với keys:
                - id, location, type, description, image_url, price
                - image_vector (List[float] - dim 512)
                - description_vector (List[float] - dim 768)

        Returns:
            Dict với keys 'text_ids' và 'image_ids'
        """
        try:
            # Validate
            for item in data:
                required_fields = ["id", "location", "type", "description",
                                   "image_url", "price", "image_vector", "description_vector"]
                for field in required_fields:
                    assert field in item, f"Missing '{field}'"

                assert len(item["image_vector"]) == self.IMAGE_VECTOR_DIM
                assert len(item["description_vector"]) == self.DESCRIPTION_VECTOR_DIM

            # Prepare common data
            ids = [item["id"] for item in data]
            locations = [item["location"] for item in data]
            types = [item["type"] for item in data]
            descriptions = [item["description"] for item in data]
            image_urls = [item["image_url"] for item in data]
            prices = [item["price"] for item in data]

            # Insert vào text collection
            text_entities = [
                ids, locations, types, descriptions, image_urls, prices,
                [item["description_vector"] for item in data]
            ]
            text_result = self.text_collection.insert(text_entities)
            self.text_collection.flush()
            logger.info(f"✅ Inserted {len(data)} records into text collection")

            # Insert vào image collection
            image_entities = [
                ids, locations, types, descriptions, image_urls, prices,
                [item["image_vector"] for item in data]
            ]
            image_result = self.image_collection.insert(image_entities)
            self.image_collection.flush()
            logger.info(f"✅ Inserted {len(data)} records into image collection")

            return {
                "text_ids": text_result.primary_keys,
                "image_ids": image_result.primary_keys
            }

        except Exception as e:
            logger.error(f"❌ Failed to insert data: {e}")
            raise

    def search_by_description(
            self,
            query_vector: List[float],
            top_k: int = 10,
            filters: Optional[str] = None
    ) -> List[Dict]:
        """Tìm kiếm bằng description vector"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = self.text_collection.search(
            data=[query_vector],
            anns_field="description_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["id", "location", "type", "description", "image_url", "price"]
        )

        return self._format_results(results)

    def search_by_image(
            self,
            query_vector: List[float],
            top_k: int = 10,
            filters: Optional[str] = None
    ) -> List[Dict]:
        """Tìm kiếm bằng image vector"""
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        results = self.image_collection.search(
            data=[query_vector],
            anns_field="image_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["id", "location", "type", "description", "image_url", "price"]
        )

        return self._format_results(results)

    def hybrid_search(
            self,
            text_vector: List[float],
            image_vector: List[float],
            text_weight: float = 0.7,
            image_weight: float = 0.3,
            top_k: int = 10,
            filters: Optional[str] = None
    ) -> List[Dict]:
        """Tìm kiếm kết hợp text và image"""
        text_results = self.search_by_description(text_vector, top_k * 2, filters)
        image_results = self.search_by_image(image_vector, top_k * 2, filters)

        # Merge results
        merged = {}

        for result in text_results:
            result_id = result["id"]
            merged[result_id] = {
                **result,
                "text_score": result["score"],
                "image_score": 0,
                "combined_score": result["score"] * text_weight
            }

        for result in image_results:
            result_id = result["id"]
            if result_id in merged:
                merged[result_id]["image_score"] = result["score"]
                merged[result_id]["combined_score"] += result["score"] * image_weight
            else:
                merged[result_id] = {
                    **result,
                    "text_score": 0,
                    "image_score": result["score"],
                    "combined_score": result["score"] * image_weight
                }

        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        return sorted_results

    def get_by_id(self, destination_id: int) -> Optional[Dict]:
        """Lấy thông tin theo ID (từ text collection)"""
        results = self.text_collection.query(
            expr=f"id == {destination_id}",
            output_fields=["id", "location", "type", "description", "image_url", "price"]
        )
        return results[0] if results else None

    def get_by_location(self, location: str, limit: int = 20) -> List[Dict]:
        """Lấy điểm du lịch theo location"""
        results = self.text_collection.query(
            expr=f'location == "{location}"',
            output_fields=["id", "location", "type", "description", "image_url", "price"],
            limit=limit
        )
        return results

    def get_statistics(self) -> Dict:
        """Thống kê collections"""
        return {
            "database": self.DATABASE_NAME,
            "text_collection": {
                "name": self.TEXT_COLLECTION_NAME,
                "count": self.text_collection.num_entities,
                "vector_dim": self.DESCRIPTION_VECTOR_DIM
            },
            "image_collection": {
                "name": self.IMAGE_COLLECTION_NAME,
                "count": self.image_collection.num_entities,
                "vector_dim": self.IMAGE_VECTOR_DIM
            }
        }

    @staticmethod
    def _format_results(results) -> List[Dict]:
        """Format kết quả search"""
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.entity.get("id"),
                    "location": hit.entity.get("location"),
                    "type": hit.entity.get("type"),
                    "description": hit.entity.get("description"),
                    "image_url": hit.entity.get("image_url"),
                    "price": hit.entity.get("price"),
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance)
                })
        return formatted

    def drop_collections(self):
        """Xóa cả 2 collections"""
        if utility.has_collection(self.TEXT_COLLECTION_NAME):
            utility.drop_collection(self.TEXT_COLLECTION_NAME)
            logger.info(f"✅ Dropped {self.TEXT_COLLECTION_NAME}")

        if utility.has_collection(self.IMAGE_COLLECTION_NAME):
            utility.drop_collection(self.IMAGE_COLLECTION_NAME)
            logger.info(f"✅ Dropped {self.IMAGE_COLLECTION_NAME}")


if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("Testing TourismMultimodalDAO with 2 separate collections")
    print("=" * 70)

    try:
        dao = TourismMultimodalDAO(host="localhost", port="19530")

        stats = dao.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"  Database: {stats['database']}")
        print(f"  Text Collection: {stats['text_collection']['name']} ({stats['text_collection']['count']} items)")
        print(f"  Image Collection: {stats['image_collection']['name']} ({stats['image_collection']['count']} items)")

        print(f"\n📝 Inserting sample data...")
        sample_data = [
            {
                "id": 1,
                "location": "Đà Nẵng",
                "type": "beach",
                "description": "Bãi biển Mỹ Khê với cát trắng mịn",
                "image_url": "https://havi-web.s3.ap-southeast-1.amazonaws.com/bien_my_khe_da_nang_2_11zon_1_a3a8e98ee1.webp",
                "price": 0.0,
                "image_vector": np.random.rand(dao.IMAGE_VECTOR_DIM).tolist(),
                "description_vector": np.random.rand(dao.DESCRIPTION_VECTOR_DIM).tolist()
            },
            {
                "id": 2,
                "location": "Hội An",
                "type": "historical",
                "description": "Phố cổ Hội An - Di sản văn hóa thế giới",
                "image_url": "https://dulichnewtour.vn/ckfinder/images/Tours/Domestic/Quang-Nam/pho-co-hoi-an(1).jpg",
                "price": 120000.0,
                "image_vector": np.random.rand(dao.IMAGE_VECTOR_DIM).tolist(),
                "description_vector": np.random.rand(dao.DESCRIPTION_VECTOR_DIM).tolist()
            }
        ]

        result = dao.insert_data(sample_data)
        print(f"✅ Inserted - Text IDs: {result['text_ids']}, Image IDs: {result['image_ids']}")

        print(f"\n🔍 Testing query by location...")
        results = dao.get_by_location("Đà Nẵng")
        print(f"✅ Found {len(results)} destinations in Đà Nẵng")
        for r in results:
            print(f"   - ID {r['id']}: {r['description']}")

        print("\n✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)