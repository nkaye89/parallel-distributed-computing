#include <iostream>
#include <stdlib.h>
#include <thread>
#include <chrono>

using namespace std;

const int MAX_ITEMS = 20000000;
int numbers[MAX_ITEMS];
int temp[MAX_ITEMS];

template<class ItemType>
void Merge(ItemType values[], int leftFirst, int leftLast,
           int rightFirst, int rightLast, ItemType tempArray[])
{
    int index = leftFirst;
    int saveFirst = leftFirst;
    while ((leftFirst <= leftLast) && (rightFirst <= rightLast))
    {
        if (values[leftFirst] < values[rightFirst]) // If left value is less, take it
        {
            tempArray[index] = values[leftFirst];
            leftFirst++;
        }
        else
        {
            tempArray[index] = values[rightFirst];  // Otherwise take right value
            rightFirst++;
        }
        index++;
    }
    // Left or right can have extra values (but not both)
    while (leftFirst <= leftLast)
        // Copy remaining items from left half.
    {
        tempArray[index] = values[leftFirst];
        leftFirst++;
        index++;
    }
    while (rightFirst <= rightLast)
        // Copy remaining items from right half.
    {
        tempArray[index] = values[rightFirst];
        rightFirst++;
        index++;
    }
    // Copy temporary array back into values
    for (index = saveFirst; index <= rightLast; index++)
        values[index] = tempArray[index];
}

template<class ItemType>
void MergeSort(ItemType values[], int first, int last, ItemType tempArray[])
{
    if (first < last) //Can still divide incoming piece (otherwise just return)
    {
        int middle = (first + last) / 2;
        MergeSort<ItemType>(values, first, middle, tempArray);    // Recursively sort lower (left) half
        MergeSort<ItemType>(values, middle + 1, last, tempArray); // Recursively sort upper (right) half
        Merge<ItemType>(values, first, middle, middle + 1, last, tempArray); // Merge sorted halves
    }
}

int main(int argc, const char * argv[]) {
    chrono::time_point<chrono::system_clock> start;
    chrono::time_point<chrono::system_clock> end;
    
    // Initialize the array with random integers
    for (int index = 0; index < MAX_ITEMS; index++) {
        numbers[index] = rand() % 1000000000;
    }
    
    start = chrono::system_clock::now();             // Record start time
    
    MergeSort<int>(numbers, 0, MAX_ITEMS-1, temp);   // Run the sort
    
    end = chrono::system_clock::now();               // Record end time
    
    chrono::duration<float> elapsed = end-start;     // Calculate and report time
    cout << "Execution time in seconds = " << elapsed.count() << "\n";
    
    return 0;
}
