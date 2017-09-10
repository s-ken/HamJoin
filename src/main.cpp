#include "type.h"
#include "Timer.h"
#include "SetTable.h"
#include "SigTable.h"
#include "functions.h"

#include <string>
#include <vector>
#include <utility>

#include <iostream>

using namespace std;

int main() {
    Timer timer;
    string file_path = "../dataset/enron/enron_10k.txt";

    cout << "Read file..." << flush;
    timer.start();
    SetTable table( file_path );
    timer.end();

    cout << "Construct signature matrix..." << flush;
    SigTable sig_mat;
    timer.start();
    sig_mat.sketch( table );
    timer.end();
    sig_mat.print_info();

    cout << "Join..." << flush;
    vector< pair< int, int > > res;
    timer.start();
    sig_mat.join( res );
    timer.end();
    cout << "# outputs = " << res.size() << endl;

    cout << "Join..." << flush;
    vector< pair< int, int > > res_naive;
    timer.start();
    table.join( res_naive );
    timer.end();
    cout << "# outputs = " << res_naive.size() << endl;

    cout << "Evaluate recall and precision..." << flush;
    timer.start();
    auto precision_recall = eval_recall_precision( res_naive, res );
    timer.end();
    cout << "Precision = " << precision_recall.first << ", Recall = " << precision_recall.second << endl;

}
