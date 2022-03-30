import pyreadr


def main():
    result = pyreadr.read_r('titanic_data.rda')

    df1 = result['titanic_data']
    df1.to_csv('titanic.csv')


if __name__ == '__main__':
    main()
