#include "app-desc.hpp"
#include "compare.hpp"

int compare(const void *p1, const void *p2)
{
     item_t *a = (item_t *) p1;
     item_t *b = (item_t *) p2;
     double c = ((double) a->value / a->weight) -
     ((double) b->value / b->weight);

     if (c > 0) return -1;
     if (c < 0) return 1;
     return 0;
}