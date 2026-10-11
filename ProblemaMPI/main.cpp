#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <ctime>
#include <chrono>

using namespace std;

const int TAG_RANGO    = 1;
const int TAG_EXITO    = 2;
const int TAG_STOP     = 3;
const int TAG_TERMINE  = 4;

class RegistroLog {
private:
    ofstream archivo;
    omp_lock_t cerrojo;

    static string marcaTiempo() {
        using namespace std::chrono;
        time_t t = system_clock::to_time_t(system_clock::now());
        tm* info = localtime(&t);
        char* buf = new char[16];
        strftime(buf, 16, "%H:%M:%S", info);
        string s(buf);
        delete[] buf;
        return s;
    }

public:
    RegistroLog(const string& nombreArchivo) {
        archivo.open(nombreArchivo.c_str());
        omp_init_lock(&cerrojo);
    }
    ~RegistroLog() {
        if (archivo.is_open()) archivo.close();
        omp_destroy_lock(&cerrojo);
    }
    void escribir(const string& mensaje) {
        omp_set_lock(&cerrojo);
        cout << mensaje << endl;
        if (archivo.is_open()) archivo << "[" << marcaTiempo() << "] " << mensaje << endl;
        omp_unset_lock(&cerrojo);
    }
};

class BusquedaClave {
public:
    static char* crearAlfabeto() {
        char* alfabeto = new char[37];
        int idx = 0;
        for (char c = 'A'; c <= 'Z'; c++) alfabeto[idx++] = c;
        for (char c = '0'; c <= '9'; c++) alfabeto[idx++] = c;
        alfabeto[36] = '\0';
        return alfabeto;
    }

    static long long calcularEspacio(int longitud) {
        long long total = 1;
        for (int i = 0; i < longitud; i++) total *= 36;
        return total;
    }

    static void indiceATexto(long long indice, int longitud, const char* alfabeto, char* destino) {
        for (int pos = longitud - 1; pos >= 0; pos--) {
            destino[pos] = alfabeto[indice % 36];
            indice /= 36;
        }
        destino[longitud] = '\0';
    }

    static bool validarClave(const string& clave) {
        if (clave.empty() || clave.size() > 10) return false;
        for (char c : clave) {
            bool esLetra = (c >= 'A' && c <= 'Z');
            bool esDigito = (c >= '0' && c <= '9');
            if (!esLetra && !esDigito) return false;
        }
        return true;
    }
};

void calcularRangoEsclavo(long long T, int E, int indiceEsclavo, long long& inicio, long long& fin) {
    long long base = T / E, resto = T % E;
    inicio = indiceEsclavo * base + (indiceEsclavo < resto ? indiceEsclavo : resto);
    long long cuenta = base + (indiceEsclavo < resto ? 1 : 0);
    fin = inicio + cuenta - 1;   // inclusive; si cuenta==0, fin < inicio (rango vacio)
}

struct Hallazgo {
    int rank;
    int hilo;
    long long indice;
    double tiempoParcial;
};

void ejecutarEsclavo(int rank, const string& equipo, RegistroLog& log, bool detallado,
                      int longitud, const char* claveObjetivo, const char* alfabeto,
                      double tiempoInicioGlobal) {
    long long inicio, fin;
    MPI_Status estado;
    long long datosRango[2];
    MPI_Recv(datosRango, 2, MPI_LONG_LONG, 0, TAG_RANGO, MPI_COMM_WORLD, &estado);
    inicio = datosRango[0];
    fin = datosRango[1];

    long long total = (fin >= inicio) ? (fin - inicio + 1) : 0;
    {
        ostringstream m;
        m << "[Equipo: " << equipo << "][Nodo MPI: " << rank << "] Inicio de busqueda. Rango asignado: ["
          << inicio << " - " << fin << "] (" << total << " combinaciones) con " << omp_get_max_threads() << " hilos.";
        log.escribir(m.str());
    }

    bool encontradoLocal = false;
    bool detenido = false;
    Hallazgo mio; mio.rank = rank; mio.hilo = -1; mio.indice = -1; mio.tiempoParcial = 0;

    long long TAMANO_SEGMENTO = total / 10;
    if (TAMANO_SEGMENTO < 1000) TAMANO_SEGMENTO = 1000;
    if (TAMANO_SEGMENTO > 200000) TAMANO_SEGMENTO = 200000;
    long long cursor = inicio;

    while (cursor <= fin && !detenido && !encontradoLocal) {
        long long finSegmento = cursor + TAMANO_SEGMENTO - 1;
        if (finSegmento > fin) finSegmento = fin;

        #pragma omp parallel for schedule(dynamic) shared(encontradoLocal, mio)
        for (long long idx = cursor; idx <= finSegmento; idx++) {
            if (encontradoLocal) continue;

            int tid = omp_get_thread_num();
            thread_local char* candidato = nullptr;
            if (candidato == nullptr) candidato = new char[longitud + 1];

            BusquedaClave::indiceATexto(idx, longitud, alfabeto, candidato);

            if (detallado) {
                ostringstream m;
                m << "[Equipo: " << equipo << "][Nodo MPI: " << rank << "][Hilo OpenMP: " << tid
                  << "][Candidato: " << candidato << "][Estado: Evaluando]";
                log.escribir(m.str());
            }

            if (strcmp(candidato, claveObjetivo) == 0) {
                #pragma omp critical(hallazgoLocal)
                {
                    if (!encontradoLocal) {
                        encontradoLocal = true;
                        mio.hilo = tid;
                        mio.indice = idx;
                        mio.tiempoParcial = MPI_Wtime() - tiempoInicioGlobal;
                    }
                }
            }
        }

        if (encontradoLocal) {
            ostringstream m;
            m << "[EXITO] [Equipo: " << equipo << "][Nodo MPI: " << rank << "][Hilo OpenMP: " << mio.hilo
              << "][Clave Encontrada: \"" << claveObjetivo << "\"][Indice: " << mio.indice
              << "][Tiempo Parcial: " << fixed << setprecision(6) << mio.tiempoParcial << "s]";
            log.escribir(m.str());

            double datosExito[3] = { (double)mio.hilo, (double)mio.indice, mio.tiempoParcial };
            MPI_Send(datosExito, 3, MPI_DOUBLE, 0, TAG_EXITO, MPI_COMM_WORLD);
            break;
        }

        int hayStop = 0;
        MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &hayStop, &estado);
        if (hayStop) {
            long long basura;
            MPI_Recv(&basura, 1, MPI_LONG_LONG, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            detenido = true;
            ostringstream m;
            m << "[Equipo: " << equipo << "][Nodo MPI: " << rank
              << "] Orden de parada recibida (otro nodo ya encontro la clave). Deteniendo busqueda.";
            log.escribir(m.str());
            break;
        }

        cursor = finSegmento + 1;
    }

    if (!encontradoLocal && !detenido) {
        ostringstream m;
        m << "[Equipo: " << equipo << "][Nodo MPI: " << rank << "] Rango completo revisado sin encontrar la clave.";
        log.escribir(m.str());
    }

    long long fin_msg = 0;
    MPI_Send(&fin_msg, 1, MPI_LONG_LONG, 0, TAG_TERMINE, MPI_COMM_WORLD);

    ostringstream m;
    m << "[Equipo: " << equipo << "][Nodo MPI: " << rank << "] Finalizacion limpia del proceso.";
    log.escribir(m.str());
}

void ejecutarMaestro(int numProcesos, const string& equipo, RegistroLog& log,
                     int longitud, const char* claveObjetivo, double tiempoInicioGlobal,
                     long long espacio, bool& huboExito, Hallazgo& ganador, double& tiempoTotal) {
    int E = numProcesos - 1;
    huboExito = false;
    ganador.rank = -1; ganador.hilo = -1; ganador.indice = -1; ganador.tiempoParcial = 0;

    for (int e = 0; e < E; e++) {
        long long rango[2];
        calcularRangoEsclavo(espacio, E, e, rango[0], rango[1]);
        int destino = e + 1;
        MPI_Send(rango, 2, MPI_LONG_LONG, destino, TAG_RANGO, MPI_COMM_WORLD);

        ostringstream m;
        m << "[Equipo: " << equipo << "][Nodo MPI: 0] Rango [" << rango[0] << " - " << rango[1]
          << "] enviado al esclavo " << destino << ".";
        log.escribir(m.str());
    }

    int terminados = 0;
    while (terminados < E) {
        MPI_Status estado;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &estado);

        if (estado.MPI_TAG == TAG_EXITO) {
            double datos[3];
            MPI_Recv(datos, 3, MPI_DOUBLE, estado.MPI_SOURCE, TAG_EXITO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (!huboExito) {
                huboExito = true;
                ganador.rank = estado.MPI_SOURCE;
                ganador.hilo = (int)datos[0];
                ganador.indice = (long long)datos[1];
                ganador.tiempoParcial = datos[2];
                tiempoTotal = MPI_Wtime() - tiempoInicioGlobal;

                for (int e = 0; e < E; e++) {
                    int destino = e + 1;
                    if (destino == ganador.rank) continue;
                    long long senal = 1;
                    MPI_Send(&senal, 1, MPI_LONG_LONG, destino, TAG_STOP, MPI_COMM_WORLD);
                }
                ostringstream m;
                m << "[Equipo: " << equipo << "][Nodo MPI: 0] Clave encontrada por el nodo " << ganador.rank
                  << ", hilo " << ganador.hilo << ". Orden de parada enviada a los demas esclavos.";
                log.escribir(m.str());
            }
        } else if (estado.MPI_TAG == TAG_TERMINE) {
            long long basura;
            MPI_Recv(&basura, 1, MPI_LONG_LONG, estado.MPI_SOURCE, TAG_TERMINE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            terminados++;
        }
    }

    if (!huboExito) tiempoTotal = MPI_Wtime() - tiempoInicioGlobal;

    ostringstream m;
    m << "[Equipo: " << equipo << "][Nodo MPI: 0] Finalizacion limpia confirmada de los " << E << " esclavos.";
    log.escribir(m.str());
}

bool busquedaSecuencial(int longitud, const char* claveObjetivo, const char* alfabeto,
                         long long espacio, long long& indiceHallado, RegistroLog& log,
                         const string& equipo) {
    char* candidato = new char[longitud + 1];
    bool encontrado = false;

    for (long long idx = 0; idx < espacio; idx++) {
        BusquedaClave::indiceATexto(idx, longitud, alfabeto, candidato);
        if (strcmp(candidato, claveObjetivo) == 0) {
            indiceHallado = idx;
            encontrado = true;
            break;
        }
    }
    delete[] candidato;

    ostringstream m;
    m << "[Equipo: " << equipo << "][Nodo MPI: 0][Hilo OpenMP: 0][Modo SECUENCIAL] "
      << (encontrado ? ("Clave encontrada en indice " + to_string(indiceHallado)) : string("Clave NO encontrada"));
    log.escribir(m.str());
    return encontrado;
}

int main(int argc, char** argv) {
    int provisto;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provisto);

    int rank, numProcesos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesos);

    char* nombreBuf = new char[MPI_MAX_PROCESSOR_NAME];
    int lon;
    MPI_Get_processor_name(nombreBuf, &lon);
    string equipo(nombreBuf);
    delete[] nombreBuf;

    ostringstream nombreLog;
    nombreLog << "log_equipo_" << equipo << "_nodo_" << rank << ".txt";
    RegistroLog log(nombreLog.str());

    {
        ostringstream h;
        h << "[Equipo: " << equipo << "][Nodo MPI: " << rank << " de " << numProcesos
          << "][Hilos OpenMP activos: " << omp_get_max_threads() << "]";
        log.escribir(h.str());
        if (rank == 0) {
            log.escribir("Avila Martinez Marco | Larios Hernandez Carlos Alberto | Macias Renteria Dante Yael | Salcedo Ramos Luis Gael");
        }
    }

    char* alfabeto = BusquedaClave::crearAlfabeto();
    char* claveObjetivo = nullptr;
    int longitud = 0;
    bool detallado = true;
    bool claveLista = false;

    double tSecuencial = -1, tDistribuido = -1;
    int opcion = -1;

    do {
        if (rank == 0) {
            cout << "\n===== BUSQUEDA DISTRIBUIDA DE CLAVES (MPI + OpenMP) =====\n";
            cout << "1. Ingresar clave personalizada (A-Z, 0-9)\n";
            cout << "2. Ejecutar busqueda SECUENCIAL (1 nodo, 1 hilo - linea base)\n";
            cout << "3. Ejecutar busqueda PARALELA DISTRIBUIDA (MPI + OpenMP)\n";
            cout << "4. Mostrar comparativa de tiempos y speedup\n";
            cout << "5. Salir\n";
            cout << "Seleccione una opcion: ";
            cout.flush();
            if (!(cin >> opcion)) opcion = 5;
        }
        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (opcion == 1) {
            string entrada;
            int valido = 0;
            if (rank == 0) {
                do {
                    cout << "Ingrese la clave (solo A-Z y 0-9, maximo 10 caracteres): ";
                    cout.flush();
                    cin >> entrada;
                    for (char& c : entrada) c = toupper(c);
                    valido = BusquedaClave::validarClave(entrada) ? 1 : 0;
                    if (!valido) cout << "Clave invalida. Debe usar solo A-Z, 0-9, y maximo 10 caracteres.\n";
                } while (!valido);

                char det = 'S';
                cout << "Mostrar impresiones detalladas por candidato (recomendado solo con 3-4 caracteres)? (S/N): ";
                cout.flush();
                cin >> det;
                detallado = (det == 'S' || det == 's');
            }

            int longBuf = 0;
            if (rank == 0) longBuf = (int)entrada.size();
            MPI_Bcast(&longBuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
            longitud = longBuf;

            delete[] claveObjetivo;
            claveObjetivo = new char[longitud + 1];
            if (rank == 0) strcpy(claveObjetivo, entrada.c_str());
            MPI_Bcast(claveObjetivo, longitud + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

            int detInt = detallado ? 1 : 0;
            MPI_Bcast(&detInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            detallado = (detInt == 1);

            claveLista = true;
            tSecuencial = -1; tDistribuido = -1;

            if (rank == 0) {
                long long espacio = BusquedaClave::calcularEspacio(longitud);
                ostringstream m;
                m << "Clave configurada: \"" << claveObjetivo << "\" (longitud " << longitud
                  << "). Espacio de busqueda: 36^" << longitud << " = " << espacio << " combinaciones.";
                log.escribir(m.str());
            }
        }

        else if (opcion == 2) {
            if (!claveLista) { if (rank == 0) cout << "Primero ingrese una clave (opcion 1).\n"; continue; }

            long long espacio = BusquedaClave::calcularEspacio(longitud);

            if (rank == 0) {
                if (espacio > 100000000LL) {
                    char continuar = 'N';
                    cout << "ADVERTENCIA: el modo secuencial (1 solo hilo) revisara hasta " << espacio
                         << " combinaciones. Esto puede tardar MUCHO tiempo. Continuar? (S/N): ";
                    cout.flush();
                    cin >> continuar;
                    if (continuar != 'S' && continuar != 's') { cout << "Operacion cancelada.\n"; continue; }
                }

                double t0 = MPI_Wtime();
                long long idxHallado = -1;
                busquedaSecuencial(longitud, claveObjetivo, alfabeto, espacio, idxHallado, log, equipo);
                tSecuencial = MPI_Wtime() - t0;

                ostringstream m;
                m << "--- SECUENCIAL finalizado --- Tiempo total (MPI_Wtime): " << fixed << setprecision(6)
                  << tSecuencial << " s";
                log.escribir(m.str());
            }
        }

        else if (opcion == 3) {
            if (!claveLista) { if (rank == 0) cout << "Primero ingrese una clave (opcion 1).\n"; continue; }
            if (numProcesos < 2) {
                if (rank == 0) cout << "Se necesitan al menos 2 procesos (1 maestro + 1 esclavo). Relance con -n mayor a 1.\n";
                continue;
            }

            long long espacio = BusquedaClave::calcularEspacio(longitud);

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            if (rank == 0) {
                bool huboExito; Hallazgo ganador; double tiempoTotal;
                ejecutarMaestro(numProcesos, equipo, log, longitud, claveObjetivo, t0, espacio, huboExito, ganador, tiempoTotal);

                tDistribuido = tiempoTotal;
                ostringstream m;
                m << "\n=== RESULTADO DISTRIBUIDO === Clave: \"" << claveObjetivo << "\" | ";
                if (huboExito) {
                    m << "ENCONTRADA por Nodo MPI " << ganador.rank << ", Hilo OpenMP " << ganador.hilo
                      << ", Indice " << ganador.indice;
                } else {
                    m << "NO ENCONTRADA en el espacio de busqueda";
                }
                m << " | Tiempo total (MPI_Wtime): " << fixed << setprecision(6) << tiempoTotal << " s"
                  << " | Procesos: " << numProcesos << " (1 maestro + " << (numProcesos - 1) << " esclavos)";
                log.escribir(m.str());
            } else {
                ejecutarEsclavo(rank, equipo, log, detallado, longitud, claveObjetivo, alfabeto, t0);
            }
        }

        else if (opcion == 4) {
            if (rank == 0) {
                log.escribir("\n===== COMPARATIVA DE TIEMPOS =====");
                if (tSecuencial > 0) log.escribir("Tiempo SECUENCIAL (1 nodo, 1 hilo): " + to_string(tSecuencial) + " s");
                else log.escribir("Secuencial: aun no ejecutado (opcion 2).");

                if (tDistribuido > 0) log.escribir("Tiempo DISTRIBUIDO (MPI + OpenMP, " + to_string(numProcesos) + " procesos): " + to_string(tDistribuido) + " s");
                else log.escribir("Distribuido: aun no ejecutado (opcion 3).");

                if (tSecuencial > 0 && tDistribuido > 0) {
                    double speedup = tSecuencial / tDistribuido;
                    ostringstream m;
                    m << fixed << setprecision(3) << "Speedup (secuencial / distribuido): " << speedup << "x";
                    log.escribir(m.str());
                }
            }
        }
        else if (opcion == 5) {
            if (rank == 0) cout << "Saliendo del programa...\n";
        }
        else {
            if (rank == 0) cout << "Opcion invalida.\n";
        }

    } while (opcion != 5);

    delete[] alfabeto;
    delete[] claveObjetivo;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        log.escribir("Avila Martinez Marco | Larios Hernandez Carlos Alberto | Macias Renteria Dante Yael | Salcedo Ramos Luis Gael");
    }

    MPI_Finalize();
    return 0;
}
