/********************************************************************************
* dense_layer.hpp: Innehåller funktionalitet för implementering av dense-lager
*                  i neurala nätverk (dolda lager och utgångslager) via
*                  strukten dense_layer.
********************************************************************************/
#ifndef DENSE_LAYER_HPP_
#define DENSE_LAYER_HPP_

/* Inkluderingsdirektiv: */
#include <iostream> /* Funktionalitet för inmatning och utskrift. */
#include <iomanip>  /* I/O-manipulationer, såsom antalet decimaler vid utskrift. */
#include <vector>   /* Dynamiska vektorer. */
#include <cstdlib>  /* Randomiseringsfunktionen rand. */

/********************************************************************************
* dense_layer: Dense-lager för implementering i neurala nätverk. Nodernas
*              parametrar (utsignal, bias, fel och vikter) lagras via vektorer.
********************************************************************************/
struct dense_layer
{
   std::vector<double> output;               /* Nodernas utsignaler. */
   std::vector<double> error;                /* Nodernas avvikelser/felvärden. */
   std::vector<double> bias;                 /* Nodernas bias/vilovärden. */
   std::vector<std::vector<double>> weights; /* Nodernas vikter. */

   /********************************************************************************
   * dense_layer: Skapar ett nytt tomt dense-lager.
   ********************************************************************************/
   dense_layer(void) { }

   /********************************************************************************
   * dense_layer: Skapar och initierar ett nytt dense-lager med angivet antal
   *              noder samt vikter per nod.
   *
   *              - num_nodes  : Antalet noder i det nya lagret.
   *              - num_weights: Antalet vikter per nod i det nya lagret.
   ********************************************************************************/
   dense_layer(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->resize(num_nodes, num_weights);
      return;
   }

   /********************************************************************************
   * ~dense_layer: Nollställer dense-lager automatiskt innan objektet raderas.
   ********************************************************************************/
   ~dense_layer(void)
   {
      this->clear();
      return;
   }

   /********************************************************************************
   * clear: Nollställer dense-lager genom att tömma vektorer.
   ********************************************************************************/
   void clear(void)
   {
      this->output.clear();
      this->error.clear();
      this->bias.clear();
      this->weights.clear();
      return;
   }

   /********************************************************************************
   * resize: Sätter nytt antal noder och vikter per nod i angivet dense-lager.
   *         Alla bias och vikter tilldelas randomiserade startvärden mellan 0 - 1,
   *         övriga parametrar sätts till 0. Funktionen fungerar enligt nedan.
   * 
   *         1. Eventuellt tidigare innehåll frigörs innan omallokeringen.
   *         2. Vektorer för lagring av utsignaler, fel samt bias sätts till att
   *            rymma flyttal för specificerat antal noder, där samtliga parametrar 
   *            tilldelas startvärdet 0.
   *         3. Den tvådimensionella vektorn weights sätts till att rymma
   *            endimensionella vektorer innehållande flyttal för specificerat
   *            antal vikter. För varje nod läggs en sådan vektor till, därav
   *            sätts antalet endimensionella vektorer till antalet noder i lagret.
   *         4. Samtliga bias-värden och vikter sätts till randomiserade 
   *            startvärden mellan 0 - 1 via iteration i kombination med anrop 
   *            av medlemsfunktionen get_random.
   * 
   *         - num_nodes  : Nytt antal noder i dense-lagret.
   *         - num_weights: Nytt antal vikter per nod i dense-lagret.
   ********************************************************************************/
   void resize(const std::size_t num_nodes,
               const std::size_t num_weights)
   {
      this->clear();

      this->output.resize(num_nodes, 0);
      this->error.resize(num_nodes, 0);
      this->bias.resize(num_nodes, 0);
      
      this->weights.resize(num_nodes, std::vector<double>(num_weights, 0));

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] = this->get_random();

         for (std::size_t j = 0; j < this->num_weights(); ++j)
         {
            this->weights[i][j] = this->get_random();
         }
      }

      return;
   }

   /********************************************************************************
   * get_random: Returnerar ett randomiserat flyttal mellan 0 - 1.
   ********************************************************************************/
   double get_random(void) 
   {
      return static_cast<double>(std::rand()) / RAND_MAX;
   }

   /********************************************************************************
   * num_nodes: Returnerar antalet noder i aktuellt dense-lager.
   ********************************************************************************/
   std::size_t num_nodes(void) const
   {
      return this->output.size();
   }

   /********************************************************************************
   * num_weights: Returnerar antalet vikter per nod i aktuellt dense-lager.
   ********************************************************************************/
   std::size_t num_weights(void) const
   {
      if (this->weights.size())
      {
         return this->weights[0].size();
      }
      else
      {
         return 0;
      }
   }

   /********************************************************************************
   * feedforward: Beräknar nya utsignaler igenom dense-lagret via nya insignaler,
   *              lagrade i en vektor. Beräkningen genomförs enligt nedan:
   * 
   *              1. Itererar genom lagret från nod till nod.
   *              2. Summan av nodens bias samt dess vikter adderas.
   *              3. Om summan överstiger 0 är noden aktiverad. Då sätts nodens
   *                 utsignal till denna summa. Annars sätts utsignalen till 0.
   * 
   *              - input: Nya insignaler till dense-lagret.
   ********************************************************************************/
   void feedforward(const std::vector<double>& input)
   {
      for (int i = 0; i < this->num_nodes(); ++i)
      {
         auto sum = this->bias[i];

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            sum += input[j] * this->weights[i][j];
         }

         if (sum > 0)
         {
            this->output[i] = sum;
         }
         else
         {
            this->output[i] = 0;
         }
      }

      return;
   }

   /********************************************************************************
   * backpropagate: Beräknar aktuella fel i utgångslagret genom att jämföra
   *                referensvärden från träningsdatan med utsignaler ur lagret.
   *                Enbart om noden är aktiverad beräknas fel. OBS! Denna funktion
   *                är enbart avsedd för utgångslager.
   * 
   *                1. Iterera från nod till nod i utgångslagret.
   *                2. Beräkna avvikelse som differensen mellan aktuellt 
   *                   referensvärde samt motsvarande utsignal.
   *                3. Om noden är aktiverad sparas detta värde som aktuellt
   *                   fel / aktuell avvikelse på noden. Annars räknas inget fel.
   * 
   *                - reference: Referensvärden från träningsdatan.                
   ********************************************************************************/
   void backpropagate(const std::vector<double>& reference)
   {
      for (std::size_t i = 0; i < this->num_nodes() && i < reference.size(); ++i)
      {
         const auto dev = reference[i] - this->output[i];

         if (this->output[i] > 0)
         {
            this->error[i] = dev;
         }
         else
         {
            this->error[i] = 0;
         }
      }
      return;
   }

   /********************************************************************************
   * backpropagate: Beräknar aktuella fel i det dolda lagret genom att summera
   *                felen i nästa lager multiplicerat med vikterna mellan noderna. 
   *                OBS! Denna funktion är enbart avsedd för dolda lager.
   * 
   *                1. Iterara genom lagret från första nod till sista nod.
   *                2. För varje iteration genom lagret, iterera från första
   *                   till sista nod i nästa lager och summera felet för denna
   *                   nod samt vikten mellan noderna.
   *                3. Om noden är aktiverad (utsignalen är över 0) så sparas det 
   *                   beräknade värdet. Annars om noden inte är aktiverad så 
   *                   sparas inte värdet (en inaktiverad nod har inte bidragit
   *                   till aktuell utsignal och har därmed inte orsakat fel).
   * 
   *                - next_layer: Referens till nästa lager.
   ********************************************************************************/
   void backpropagate(const dense_layer& next_layer)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         auto dev = 0.0;

         for (std::size_t j = 0; j < next_layer.num_nodes(); ++j)
         {
            dev += next_layer.error[j] * next_layer.weights[j][i];
         }

         if (this->output[i] > 0)
         {
            this->error[i] = dev; 
         }
         else
         {
            this->error[i] = 0;
         }
      }

      return;
   }

   /********************************************************************************
   * optimize: Justerar bias och vikter i aktuellt dense-lager utefter beräknade
   *           fel samt angiven lärhastighet.
   * 
   *           1.  Iterera genom lagret från första till sista nod.
   *           2.  Justera aktuellt nods bias genom att addera aktuellt fel
   *               multiplicerat med lärhastigheten.
   *           3.  Iterera genom nodens vikter från första till sista och addera
   *               aktuellt fel multiplicerat med lärhastigheten gånger insignalen.
   * 
   *           - input:         Referens till vektor innehållande lagrets insignaler
   *                            (kan vara utsignaler från föregående lager eller
   *                            från ingångslagret).
   *           - learning_rate: Lärhastigheten, avgör justeringsgraden vid fel.
   ********************************************************************************/
   void optimize(const std::vector<double>& input,
                 const double learning_rate)
   {
      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         this->bias[i] += this->error[i] * learning_rate;

         for (std::size_t j = 0; j < this->num_weights() && j < input.size(); ++j)
         {
            this->weights[i][j] += this->error[i] * learning_rate * input[j];
         }
      }
      return;
   }

   /********************************************************************************
   * print: Skriver ut information om aktuellt dense-lager via angiven utström,
   *        där standardutenheten std::cout används som default för utskrift
   *        i terminalen.
   * 
   *        - ostream: Referens till angiven utström (default = std::cout).
   ********************************************************************************/
   void print(std::ostream& ostream = std::cout)
   {
      ostream << "--------------------------------------------------------------------------------\n";

      ostream << "Number of nodes: " << this->num_nodes() << "\n";
      ostream << "Number of weights per node: " << this->num_weights() << "\n\n";

      ostream << "Output: ";
      this->print_line(this->output, ostream);

      ostream << "Error: ";
      this->print_line(this->error, ostream);

      ostream << "Bias: ";
      this->print_line(this->bias, ostream);

      ostream << "\nWeights:\n";

      for (std::size_t i = 0; i < this->num_nodes(); ++i)
      {
         ostream << "\tNode " << i + 1 << ": ";
         this->print_line(this->weights[i], ostream);
      }
      
      ostream << "--------------------------------------------------------------------------------\n\n";
      return;
   }

   /********************************************************************************
   * print_line: Skriver ut flyttal ur en vektor på en enda rad och avslutas med
   *             en ny rad. Varje flyttal skrivs ut med två decimalers precision.
   * 
   *             - data   : Referens till vektorn vars innehåll ska skrivas ut.
   *             - ostream: Referens till den utström som utskrift ska ske till.
   ********************************************************************************/
   void print_line(const std::vector<double>& data, 
                   std::ostream& ostream)
   {
      for (auto& i : data)
      {
         ostream << std::setprecision(2) << i << " ";
      }

      ostream << "\n";
      return;
   }
};

#endif /* DENSE_LAYER_HPP_ */