#include "btor2parser/btor2parser.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t close_input;
static FILE* input_file;
static const char* input_name;

/* Parse BTOR2 file and count the keywords. */

#define NUM_KEYWORDS 69

typedef struct {
    const char *keyword;
    int id;
} Keyword;

const Keyword KEYWORDS[NUM_KEYWORDS] = {
    {"add", 0}, {"and", 1}, {"bad", 2},
    {"constraint", 3}, {"concat", 4}, {"const", 5},
    {"constd", 6}, {"consth", 7}, {"dec", 8},
    {"eq", 9}, {"fair", 10}, {"iff", 11},
    {"implies", 12}, {"iff", 13}, {"inc", 14},
    {"init", 15}, {"input", 16}, {"ite", 17},
    {"justice", 18}, {"mul", 19}, {"nand", 20},
    {"neq", 21}, {"neg", 22}, {"next", 23},
    {"nor", 24}, {"not", 25}, {"one", 26},
    {"ones", 27}, {"or", 28}, {"output", 29},
    {"read", 30}, {"redand", 31}, {"redor", 32},
    {"redxor", 33}, {"rol", 34}, {"ror", 35},
    {"saddo", 36}, {"sext", 37}, {"sgt", 38},
    {"sgte", 39}, {"sdiv", 40}, {"sdivo", 41},
    {"slice", 42}, {"sll", 43}, {"slt", 44},
    {"slte", 45}, {"sort", 46}, {"smod", 47},
    {"smulo", 48}, {"ssubo", 49}, {"sra", 50},
    {"srl", 51}, {"srem", 52}, {"state", 53},
    {"sub", 54}, {"uaddo", 55}, {"udiv", 56},
    {"uext", 57}, {"ugt", 58}, {"ugte", 59},
    {"ult", 60}, {"ulte", 61}, {"umulo", 62},
    {"urem", 63}, {"usubo", 64}, {"write", 65},
    {"xnor", 66}, {"xor", 67}, {"zero", 68}
};

int get_keyword_id(const char* keyword, const Keyword keywords[], int num_keywords);

int32_t
main (int32_t argc, char** argv)
{
  Btor2Parser* reader;
  Btor2LineIterator it;
  Btor2Line* l;
  int32_t i, verbosity = 0;
  const char* err;
  for (i = 1; i < argc; i++)
  {
    if (!strcmp (argv[i], "-h"))
    {
      fprintf (stderr, "usage: catbtor [-h|-v] [ <btorfile> ]\n");
      exit (1);
    }
    else if (!strcmp (argv[i], "-v"))
      verbosity++;
    else if (argv[i][0] == '-')
    {
      fprintf (
          stderr, "*** catbtor: invalid option '%s' (try '-h')\n", argv[i]);
      exit (1);
    }
    else if (input_name)
    {
      fprintf (stderr, "*** catbtor: too many inputs (try '-h')\n");
      exit (1);
    }
    else
      input_name = argv[i];
  }
  if (!input_name)
  {
    input_file = stdin;
    assert (!close_input);
    input_name = "<stdin>";
  }
  else
  {
    input_file = fopen (input_name, "r");
    if (!input_file)
    {
      fprintf (
          stderr, "*** can not open '%s' for reading\n", input_name);
      exit (1);
    }
    close_input = 1;
  }
  if (verbosity)
  {
    fprintf (stderr,
             "; [catbor] simple CAT for BTOR files\n"
             "; [catbor] reading '%s'\n",
             input_name);
    fflush (stderr);
  }
  reader = btor2parser_new ();
  if (!btor2parser_read_lines (reader, input_file))
  {
    err = btor2parser_error (reader);
    assert (err);
    fprintf (stderr, "*** parse error in '%s' %s\n", input_name, err);
    btor2parser_delete (reader);
    if (close_input) fclose (input_file);
    exit (1);
  }
  if (close_input) fclose (input_file);
  if (verbosity)
  {
    fprintf (stderr, "; [catbor] finished parsing '%s'\n", input_name);
    fflush (stderr);
  }
  if (verbosity)
  {
    fprintf (stderr, "; [catbor] starting to dump BTOR model to '<stdout>'\n");
    fflush (stderr);
  }
  it = btor2parser_iter_init (reader);
  int counters[NUM_KEYWORDS] = {0};
  int bits[NUM_KEYWORDS] = {0};
  // unsigned int state_bit_width_sum = 0;
  // unsigned int input_bit_width_sum = 0;
  while ((l = btor2parser_iter_next (&it)))
  {
    // printf ("%" PRId64 " %s", l->id, l->name);
    int keyword_id = -1;
    int bit_width;
    // if (l->tag == BTOR2_TAG_sort)
    // {
    //   // printf (" %s", l->sort.name);
    //   bit_width = l->sort.bitvec.width;
    //   switch (l->sort.tag)
    //   {
    //     case BTOR2_TAG_SORT_bitvec: 
    //       // printf (" %u", l->sort.bitvec.width); break;
    //       keyword_id = get_keyword_id("sort_bitvec", KEYWORDS, NUM_KEYWORDS);
    //       break;
    //     case BTOR2_TAG_SORT_array:
    //       // printf (" %" PRId64 " %" PRId64, l->sort.array.index, l->sort.array.element);
    //       keyword_id = get_keyword_id("sort_array", KEYWORDS, NUM_KEYWORDS);
    //       break;
    //     default:
    //       assert (0);
    //       fprintf (stderr, "*** invalid sort encountered\n");
    //       exit (1);
    //   }
    // }
    // else
    // {
      // if (l->tag == BTOR2_TAG_state)
      // {
      //   assert (l->sort.tag == BTOR2_TAG_SORT_bitvec);
      //   state_bit_width_sum += l->sort.bitvec.width;
      //   // printf("state bit width: %u\n", state_bit_width_sum);

      // } else if (l->tag == BTOR2_TAG_input)
      // {
      //   assert (l->sort.tag == BTOR2_TAG_SORT_bitvec);
      //   input_bit_width_sum += l->sort.bitvec.width;
      //   // printf("input bit width: %u\n", input_bit_width_sum);
      // }
    keyword_id = get_keyword_id(l->name, KEYWORDS, NUM_KEYWORDS);
    if (l->tag == BTOR2_TAG_bad || l->tag == BTOR2_TAG_constraint \
        || l->tag == BTOR2_TAG_fair || l->tag == BTOR2_TAG_output \
        || l->tag == BTOR2_TAG_justice)
    {
      bit_width = 1;
    } else 
    {
      assert (l->sort.tag == BTOR2_TAG_SORT_bitvec);
      bit_width = l->sort.bitvec.width;
    }
    if (keyword_id != -1)
    {
      counters[keyword_id]++;
      bits[keyword_id] += bit_width;
    } else {
      fprintf(stderr, "Keyword not found: %s\n", l->name);
      exit (1);
    }
  }
  btor2parser_delete (reader);
  for (int i = 0; i < NUM_KEYWORDS; i++) {
        printf("%d ", counters[i]);
  }
  printf("\n");
  for (int i = 0; i < NUM_KEYWORDS; i++) {
        printf("%d ", bits[i]);
  }
  printf("\n");
  return 0;
}

// Definition of the get_keyword_id function
int get_keyword_id(const char* keyword, const Keyword keywords[], int num_keywords) {
    for (int i = 0; i < num_keywords; i++) {
        // printf("Comparing %s with %s\n", keyword, keywords[i].keyword);
        if (strcmp(keyword, keywords[i].keyword) == 0) {
            return keywords[i].id;
        }
    }
    return -1; // Keyword not found
}