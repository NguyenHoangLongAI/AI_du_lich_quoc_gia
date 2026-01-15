"""
Script để crawl dữ liệu và insert vào Milvus database
"""

from crawler_baichay import BaiChayCrawler
from tourism_dao import BaiChayTourismDAO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("Bãi Cháy Tourism - Crawl & Insert to Milvus")
    print("=" * 80)

    # Config
    MAX_ITEMS_PER_CATEGORY = None  # None = crawl tất cả items
    MAX_PAGES_PER_CATEGORY = 12  # Số trang tối đa mỗi category
    USE_SAFE_METHOD = True  # True = crawl từng trang an toàn (khuyến nghị)

    CATEGORIES_TO_CRAWL = [
        "diem-den",
        "luu-tru",
        "tour",
        "nha-hang",
        "am-thuc",
        "du-thuyen"
    ]

    try:
        # Step 1: Khởi tạo crawler
        logger.info("\n📡 Initializing crawler...")
        crawler = BaiChayCrawler()

        # Step 2: Khởi tạo DAO
        logger.info("🗄️  Initializing database connection...")
        dao = BaiChayTourismDAO(host="localhost", port="19530")

        # Step 3: Crawl và insert từng category
        total_inserted = 0
        id_counter = 1  # ID counter để tạo unique IDs

        for category in CATEGORIES_TO_CRAWL:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"🚀 Processing category: {category}")
                logger.info(f"{'='*80}")

                # Crawl data
                items = crawler.crawl_category(
                    category,
                    max_items=MAX_ITEMS_PER_CATEGORY,
                    max_pages=MAX_PAGES_PER_CATEGORY,
                    use_safe_method=USE_SAFE_METHOD
                )

                if not items:
                    logger.warning(f"⚠️  No items crawled for {category}")
                    continue

                # Gán IDs unique
                for item in items:
                    item["id"] = id_counter
                    id_counter += 1

                # Insert vào database
                logger.info(f"\n💾 Inserting {len(items)} items into database...")
                inserted_ids = dao.insert_data(items)
                total_inserted += len(inserted_ids)

                logger.info(f"✅ Successfully inserted {len(inserted_ids)} items from {category}")

                # Save to JSON backup
                import json
                backup_file = f"backup_{category}.json"
                items_without_vectors = [
                    {k: v for k, v in item.items() if k != "description_vector"}
                    for item in items
                ]
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(items_without_vectors, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Backup saved to {backup_file}")

            except Exception as e:
                logger.error(f"❌ Error processing {category}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Step 4: Hiển thị statistics
        logger.info(f"\n{'='*80}")
        logger.info("📊 Final Statistics")
        logger.info(f"{'='*80}")

        stats = dao.get_statistics()
        logger.info(f"Database: {stats['database']}")
        logger.info(f"Collection: {stats['collection']['name']}")
        logger.info(f"Total items in DB: {stats['collection']['total_count']}")
        logger.info(f"Vector dimension: {stats['collection']['vector_dim']}")
        logger.info(f"Items inserted in this run: {total_inserted}")

        # Test search
        logger.info(f"\n{'='*80}")
        logger.info("🧪 Testing Search Functions")
        logger.info(f"{'='*80}")

        # Test search by type
        for category in CATEGORIES_TO_CRAWL:
            results = dao.search_by_type(category, limit=3)
            logger.info(f"\n{category.upper()}: {len(results)} items")
            for r in results[:2]:  # Show first 2
                logger.info(f"  • {r['name']} - {r['price_range']}")

        logger.info(f"\n{'='*80}")
        logger.info("✅ All operations completed successfully!")
        logger.info(f"{'='*80}")

    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
