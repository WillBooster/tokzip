import { jaJpModule } from '../generated/jaJp.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(jaJpModule);

export { jaJpModule };
