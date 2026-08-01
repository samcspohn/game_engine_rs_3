1. host_wait_compute increases when screen size increases. TRS staging should only wait on the scatter compute shader, not the rendering. this is a regression/bug - fixed. cross numa transfer pays 2.5x penalty. writing from the second node to the gpu or pulling from the gpu to the second node both pay this penalty. speed is achieved by keeping data in node 0's cache

tone mapping/hdr
