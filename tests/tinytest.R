if (requireNamespace("tinytest", quietly = TRUE)) {
  Sys.setenv(R_USER_CACHE_DIR = tempfile("n3d_cache_"),
             R_USER_DATA_DIR = tempfile("n3d_data_"),
             R_USER_CONFIG_DIR = tempfile("n3d_config_"))
  tinytest::test_package("n3d")
}
