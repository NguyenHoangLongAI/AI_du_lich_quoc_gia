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


class BaiChayTourismDAO:
    """DAO cho du lịch Bãi Cháy - Quảng Ninh với collection duy nhất"""

    DATABASE_NAME = "bai_chay_tourism_db"
    COLLECTION_NAME = "bai_chay_data"

    DESCRIPTION_VECTOR_DIM = 768

    def __init__(self, host="localhost", port="19530"):
        """Khởi tạo connection và tạo collection"""
        self.host = host
        self.port = port
        self.connect()
        self.switch_database()
        self.collection = self._get_or_create_collection()

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
        """Chuyển sang database bai_chay_tourism_db"""
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

    def _create_schema(self) -> CollectionSchema:
        """
        Schema cho Bãi Cháy tourism collection
        Hỗ trợ nhiều loại: điểm đến, lưu trú, tour, nhà hàng, ẩm thực, du thuyền
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
            # diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen
            FieldSchema(name="sub_type", dtype=DataType.VARCHAR, max_length=200),
            # Du lịch biển, Khách sạn cao cấp, etc.
            FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="address", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65000),
            FieldSchema(name="price_range", dtype=DataType.VARCHAR, max_length=200),
            # "Miễn phí", "350.000 - 600.000 VNĐ"
            FieldSchema(name="price_min", dtype=DataType.FLOAT),  # Giá tối thiểu (0 nếu miễn phí)
            FieldSchema(name="price_max", dtype=DataType.FLOAT),  # Giá tối đa
            FieldSchema(name="opening_hours", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="image_urls", dtype=DataType.VARCHAR, max_length=5000),  # JSON array string của nhiều URLs
            FieldSchema(name="rating", dtype=DataType.FLOAT),  # 0-5
            FieldSchema(name="view_count", dtype=DataType.INT64),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(
                name="description_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.DESCRIPTION_VECTOR_DIM
            )
        ]

        return CollectionSchema(
            fields=fields,
            description="Bãi Cháy tourism unified collection",
            enable_dynamic_field=True
        )

    def _get_or_create_collection(self) -> Collection:
        """Tạo hoặc load collection"""
        if utility.has_collection(self.COLLECTION_NAME):
            logger.info(f"✅ Collection '{self.COLLECTION_NAME}' exists, loading...")
            collection = Collection(self.COLLECTION_NAME)
        else:
            logger.info(f"🔨 Creating collection '{self.COLLECTION_NAME}'")
            schema = self._create_schema()
            collection = Collection(name=self.COLLECTION_NAME, schema=schema)

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 256}
            }
            collection.create_index(field_name="description_vector", index_params=index_params)
            logger.info("  ✅ Created index for description_vector (COSINE)")

        collection.load()
        logger.info(f"✅ Collection loaded")
        return collection

    def insert_data(self, data: List[Dict]) -> List[int]:
        """
        Chèn dữ liệu vào collection

        Args:
            data: List các dict với keys:
                - id, name, type, sub_type, location, address, description
                - price_range, price_min, price_max, opening_hours
                - image_urls (string JSON array), rating, view_count, url
                - description_vector (List[float] - dim 768)

        Returns:
            List của primary keys
        """
        try:
            # Validate
            for item in data:
                required_fields = ["id", "name", "type", "description", "description_vector"]
                for field in required_fields:
                    assert field in item, f"Missing '{field}'"
                assert len(item["description_vector"]) == self.DESCRIPTION_VECTOR_DIM

            # Prepare data
            entities = [
                [item["id"] for item in data],
                [item["name"] for item in data],
                [item["type"] for item in data],
                [item.get("sub_type", "") for item in data],
                [item.get("location", "Bãi Cháy, Quảng Ninh") for item in data],
                [item.get("address", "") for item in data],
                [item["description"] for item in data],
                [item.get("price_range", "") for item in data],
                [item.get("price_min", 0.0) for item in data],
                [item.get("price_max", 0.0) for item in data],
                [item.get("opening_hours", "") for item in data],
                [item.get("image_urls", "[]") for item in data],
                [item.get("rating", 0.0) for item in data],
                [item.get("view_count", 0) for item in data],
                [item.get("url", "") for item in data],
                [item["description_vector"] for item in data]
            ]

            result = self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"✅ Inserted {len(data)} records into collection")

            return result.primary_keys

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
            "params": {"nprobe": 20}
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="description_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["id", "name", "type", "sub_type", "location", "address",
                           "description", "price_range", "price_min", "price_max",
                           "opening_hours", "image_urls", "rating", "view_count", "url"]
        )

        return self._format_results(results)

    def search_by_type(
            self,
            tourism_type: str,
            limit: int = 20
    ) -> List[Dict]:
        """
        Tìm kiếm theo loại
        Args:
            tourism_type: diem-den, luu-tru, tour, nha-hang, am-thuc, du-thuyen
        """
        results = self.collection.query(
            expr=f'type == "{tourism_type}"',
            output_fields=["id", "name", "type", "sub_type", "location", "address",
                           "description", "price_range", "price_min", "price_max",
                           "opening_hours", "image_urls", "rating", "view_count", "url"],
            limit=limit
        )
        return results

    def get_by_id(self, item_id: int) -> Optional[Dict]:
        """Lấy thông tin theo ID"""
        results = self.collection.query(
            expr=f"id == {item_id}",
            output_fields=["id", "name", "type", "sub_type", "location", "address",
                           "description", "price_range", "price_min", "price_max",
                           "opening_hours", "image_urls", "rating", "view_count", "url"]
        )
        return results[0] if results else None

    def get_statistics(self) -> Dict:
        """Thống kê collection"""
        return {
            "database": self.DATABASE_NAME,
            "collection": {
                "name": self.COLLECTION_NAME,
                "total_count": self.collection.num_entities,
                "vector_dim": self.DESCRIPTION_VECTOR_DIM
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
                    "name": hit.entity.get("name"),
                    "type": hit.entity.get("type"),
                    "sub_type": hit.entity.get("sub_type"),
                    "location": hit.entity.get("location"),
                    "address": hit.entity.get("address"),
                    "description": hit.entity.get("description"),
                    "price_range": hit.entity.get("price_range"),
                    "price_min": hit.entity.get("price_min"),
                    "price_max": hit.entity.get("price_max"),
                    "opening_hours": hit.entity.get("opening_hours"),
                    "image_urls": hit.entity.get("image_urls"),
                    "rating": hit.entity.get("rating"),
                    "view_count": hit.entity.get("view_count"),
                    "url": hit.entity.get("url"),
                    "distance": hit.distance,
                    "score": 1 / (1 + hit.distance)
                })
        return formatted

    def drop_collection(self):
        """Xóa collection"""
        if utility.has_collection(self.COLLECTION_NAME):
            utility.drop_collection(self.COLLECTION_NAME)
            logger.info(f"✅ Dropped {self.COLLECTION_NAME}")


if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("Testing BaiChayTourismDAO")
    print("=" * 70)

    try:
        dao = BaiChayTourismDAO(host="localhost", port="19530")

        stats = dao.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"  Database: {stats['database']}")
        print(f"  Collection: {stats['collection']['name']} ({stats['collection']['total_count']} items)")

        print(f"\n📝 Inserting sample data...")
        sample_data = [
            {
                "id": 1,
                "name": "Sun World Halong Park",
                "type": "diem-den",
                "sub_type": "Công viên giải trí",
                "location": "Bãi Cháy, Quảng Ninh",
                "address": "Đường Hạ Long, phường Bãi Cháy, tỉnh Quảng Ninh",
                "description": "Sun World Halong Park là tổ hợp vui chơi giải trí lớn nhất miền Bắc với Dragon Park và Typhoon Water Park",
                "price_range": "350.000 - 600.000 VNĐ",
                "price_min": 350000.0,
                "price_max": 600000.0,
                "opening_hours": "8:00 - 22:00",
                "image_urls": '["https://duan-sungroup.com/wp-content/uploads/2022/10/thang-3-den-thang-11-la-thoi-gian-hop-ly-nhat-de-du-lich-ha-long.jpg"]',
                "rating": 4.5,
                "view_count": 62,
                "url": "https://dulichbaichay.vtcnetviet.com/sun-world-halong-park-thien-duong-giai-tri-hang-dau-tai-ha-long/",
                "description_vector": np.random.rand(dao.DESCRIPTION_VECTOR_DIM).tolist()
            }
        ]

        result = dao.insert_data(sample_data)
        print(f"✅ Inserted IDs: {result}")

        print(f"\n🔍 Testing query by type...")
        results = dao.search_by_type("diem-den")
        print(f"✅ Found {len(results)} destinations")
        for r in results:
            print(f"   - ID {r['id']}: {r['name']}")

        print("\n✅ All tests passed!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 70)