add_library(btor2parser
  btor2parser/btor2parser.c
)
target_include_directories(btor2parser PRIVATE .)
set_target_properties(btor2parser PROPERTIES PUBLIC_HEADER btor2parser/btor2parser.h)
install(
  TARGETS btor2parser
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_executable(counts counts.c)
target_link_libraries(counts btor2parser)
install(TARGETS counts DESTINATION ${CMAKE_INSTALL_BINDIR})
