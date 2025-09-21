from ocs_ci.ocs.ui.base_ui import logger
from ocs_ci.ocs.ui.page_objects.page_navigator import PageNavigator


class DataFoundationOverview(PageNavigator):
    """
    Class to represent the Data Foundation Overview page and its functionalities.
    Available starting from ODF 4.20.

    Navigation: PageNavigator / Data Foundation
    Available active links to View Storage, View Buckets, Activity monitor
    """

    def __init__(self):
        super().__init__()

    def navigate_to_view_storage(self):
        """
        Navigate to Storage Cluster page via View Storage link.

        Returns:
            StorageCluster: StorageCluster page object
        """
        logger.info("Navigate to Storage Cluster page via View Storage link")
        self.do_click(
            self.data_foundation_overview["view_storage_link"], enable_screenshot=True
        )

        from ocs_ci.ocs.ui.page_objects.storage_cluster import StorageClusterPage

        return StorageClusterPage()

    def navigate_to_view_buckets(self):
        """
        Navigate to Buckets page via View Buckets link.

        Returns:
            BucketsPage: Buckets page object
        """
        logger.info("Navigate to Buckets page via View Buckets link")
        self.do_click(
            self.data_foundation_overview["view_buckets_link"], enable_screenshot=True
        )

        from ocs_ci.ocs.ui.page_objects.object_storage import ObjectStorage

        return ObjectStorage()

    def available_vs_used_capacity_present(self):
        """
        Check if Available vs Used Capacity panel is present on the Data Foundation Overview page.

        Returns:
            bool: True if the panel is present, False otherwise.
        """
        logger.info("Check if Available vs Used Capacity panel is present")

        return self.wait_for_element_to_be_visible(
            self.data_foundation_overview["used_capacity_legend"]
        ) and self.wait_for_element_to_be_visible(
            self.data_foundation_overview["available_capacity_legend"]
        )
