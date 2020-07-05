import csv

if __name__ =="__main__":

	with open("data/RKI_COVID19.csv") as csvfile:
		with open("data/RKI_COVID19_filtered.csv", "w") as filtered_file:
			csv_reader = csv.reader(csvfile)
			csv_writer = csv.writer(filtered_file)
			for row in csv_reader:
				indices = list(range(18))
				indices.remove(3)
				indices.remove(6)
				indices.remove(8)
				indices.remove(9)

				count = 0
				for index in indices:
					index -= count
					row.pop(index)
					count += 1
				if row[3][:2] == "11":
					row[3] = "11000"
				elif row[3] == "03159":
					row[3] = "03152"
				
				row[2] = row[2].split(" ")[0]

				csv_writer.writerow(row)