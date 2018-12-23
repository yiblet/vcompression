#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define whenfail(x, y)                                                         \
    if ((x)) {                                                                 \
        y;                                                                     \
        exit(1);                                                               \
    }

int main(int argc, char *argv[]) {
    char buffer[10000];
    char *dir = argv[1];

    DIR *dp;
    struct dirent *ep;
    dp = opendir(dir);
    int bucket_count = 0;
    int in_bucket = 0;

    sprintf(buffer, "%s/bucket_%d", dir, bucket_count);
    whenfail(mkdir(buffer, S_IRWXU | S_IRGRP | S_IWGRP | S_IROTH), {
        printf("couldn't construct %s\n", buffer);
        closedir(dp);
    });

    while ((ep = readdir(dp))) {
        char old_path[10000];
        sprintf(old_path, "%s/%s", dir, ep->d_name);
        sprintf(buffer, "%s/bucket_%d/%s", dir, bucket_count, ep->d_name);
        rename(old_path, buffer);
        if (++in_bucket >= 100) {
            in_bucket = 0;
            sprintf(buffer, "%s/bucket_%d", dir, ++bucket_count);
            whenfail(mkdir(buffer, S_IRWXU | S_IRGRP | S_IWGRP | S_IROTH), {
                printf("couldn't construct new bucket: %s failed to be made\n",
                       buffer);
                closedir(dp);
            });
        }
    }

    sprintf(buffer, "%s/bucket_%d/", dir, bucket_count);
    closedir((dp));
    dp = opendir(buffer);
    if (!readdir(dp)) {
        closedir(dp);
        remove(buffer);
    } else {
        closedir(dp);
    }
}
